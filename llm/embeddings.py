import hashlib
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice
from typing import Any, cast

from sqlite_utils import Database
from sqlite_utils.db import Table

from .embeddings_migrations import embeddings_migrations
from .models import EmbeddingModel


@dataclass
class Entry:
    id: str
    score: float | None
    content: str | None = None
    metadata: dict[str, Any] | None = None
    model: str | None = None


class Collection:
    class DoesNotExist(Exception):
        pass

    def __init__(
        self,
        name: str,
        db: Database | None = None,
        *,
        model: EmbeddingModel | None = None,
        model_id: str | None = None,
        create: bool = True,
    ) -> None:
        """
        A collection of embeddings

        Returns the collection with the given name, creating it if it does not exist.

        If you set create=False a Collection.DoesNotExist exception will be raised if the
        collection does not already exist.

        Args:
            db (sqlite_utils.Database): Database to store the collection in
            name (str): Name of the collection
            model (llm.models.EmbeddingModel, optional): Embedding model to use
            model_id (str, optional): Alternatively, ID of the embedding model to use
            create (bool, optional): Whether to create the collection if it does not exist
        """
        import llm

        self.db = db or Database(memory=True)
        self.name = name
        self._model = model

        embeddings_migrations.apply(self.db)

        rows = list(self.db["collections"].rows_where("name = ?", [self.name]))
        if rows:
            row = rows[0]
            self.id = row["id"]
            self.model_id = row["model"]
        else:
            if create:
                # Collection does not exist, so model or model_id is required
                if not model and not model_id:
                    raise ValueError(
                        "Either model= or model_id= must be provided when creating a new collection"
                    )
                # Create it
                if model_id:
                    # Resolve alias
                    model = llm.get_embedding_model(model_id)
                    self._model = model
                model_id = cast(EmbeddingModel, model).model_id
                self.model_id = model_id
                self.id = (
                    cast(Table, self.db["collections"])
                    .insert(
                        {
                            "name": self.name,
                            "model": model_id,
                        }
                    )
                    .last_pk
                )
            else:
                raise self.DoesNotExist(f"Collection '{name}' does not exist")

    def model(self) -> EmbeddingModel:
        "Return the embedding model used by this collection"
        import llm

        if self._model is None:
            self._model = llm.get_embedding_model(self.model_id)

        return cast(EmbeddingModel, self._model)

    def _resolve_model(self, model: EmbeddingModel | str | None) -> EmbeddingModel:
        """
        Resolve an embedding model argument to an EmbeddingModel instance.

        None means the collection's configured model; a string is treated as
        a model ID or alias and resolved through the plugin registry.
        """
        if model is None:
            return self.model()
        if isinstance(model, EmbeddingModel):
            return model
        import llm

        return llm.get_embedding_model(model)

    def count(self) -> int:
        """
        Count the number of items in the collection.

        Returns:
            int: Number of items in the collection
        """
        return next(
            self.db.query(
                """
            select count(*) as c from embeddings where collection_id = (
                select id from collections where name = ?
            )
            """,
                (self.name,),
            )
        )["c"]

    def embed(
        self,
        id: str,
        value: str | bytes,
        metadata: dict[str, Any] | None = None,
        store: bool = False,
        *,
        key: str | None = None,
        model: EmbeddingModel | str | None = None,
    ) -> None:
        """
        Embed value and store it in the collection with a given ID.

        Args:
            id (str): ID for the value
            value (str or bytes): value to be embedded
            metadata (dict, optional): Metadata to be stored
            store (bool, optional): Whether to store the value in the content or content_blob column
            key (str, optional): API key or stored key alias to use
            model (EmbeddingModel or str, optional): Model to embed with,
                overriding the collection's configured model. The stored row
                is tagged with that model's ID. Pass a different model to
                re-index just this ID for a new generation.
        """
        from llm import encode

        embedding_model = self._resolve_model(model)
        model_id = embedding_model.model_id
        content_hash = self.content_hash(value)
        # Only skip if this exact content has already been embedded by this
        # same model generation - a different model means the row is stale.
        if self.db["embeddings"].count_where(
            "content_hash = ? and collection_id = ? and model = ?",
            [content_hash, self.id, model_id],
        ):
            return
        embedding = embedding_model.embed(value, key=key)
        cast(Table, self.db["embeddings"]).insert(
            {
                "collection_id": self.id,
                "id": id,
                "embedding": encode(embedding),
                "model": model_id,
                "content": value if (store and isinstance(value, str)) else None,
                "content_blob": value if (store and isinstance(value, bytes)) else None,
                "content_hash": content_hash,
                "metadata": json.dumps(metadata) if metadata else None,
                "updated": int(time.time()),
            },
            replace=True,
        )

    def embed_multi(
        self,
        entries: Iterable[tuple[str, str | bytes]],
        store: bool = False,
        batch_size: int = 100,
        *,
        key: str | None = None,
        model: EmbeddingModel | str | None = None,
    ) -> None:
        """
        Embed multiple texts and store them in the collection with given IDs.

        Args:
            entries (iterable): Iterable of (id: str, text: str) tuples
            store (bool, optional): Whether to store the text in the content column
            batch_size (int, optional): custom maximum batch size to use
            key (str, optional): API key or stored key alias to use
            model (EmbeddingModel or str, optional): Model to embed with,
                overriding the collection's configured model. Rows already
                embedded by that model (with unchanged content) are skipped;
                rows from other model generations are replaced in place.
        """
        self.embed_multi_with_metadata(
            ((id, value, None) for id, value in entries),
            store=store,
            batch_size=batch_size,
            key=key,
            model=model,
        )

    def embed_multi_with_metadata(
        self,
        entries: Iterable[tuple[str, str | bytes, dict[str, Any] | None]],
        store: bool = False,
        batch_size: int = 100,
        *,
        key: str | None = None,
        model: EmbeddingModel | str | None = None,
    ) -> None:
        """
        Embed multiple values along with metadata and store them in the collection with given IDs.

        Args:
            entries (iterable): Iterable of (id: str, value: str or bytes, metadata: None or dict)
            store (bool, optional): Whether to store the value in the content or content_blob column
            batch_size (int, optional): custom maximum batch size to use
            key (str, optional): API key or stored key alias to use
            model (EmbeddingModel or str, optional): Model to embed with,
                overriding the collection's configured model. Only rows that
                are missing, or were produced by a different model
                generation (or have changed content), are embedded; every
                batch is committed atomically, so an interrupted or failed
                run can safely be re-run and will converge.
        """
        import llm

        embedding_model = self._resolve_model(model)
        model_id = embedding_model.model_id
        batch_size = min(batch_size, (embedding_model.batch_size or batch_size))
        iterator = iter(entries)
        collection_id = self.id
        while True:
            batch = list(islice(iterator, batch_size))
            if not batch:
                break
            # Calculate hashes first
            items_and_hashes = [(item, self.content_hash(item[1])) for item in batch]
            # Skip IDs whose content is already embedded by this same model
            existing_ids = [
                row["id"]
                for row in self.db.query(
                    """
                    select id from embeddings
                    where collection_id = ? and model = ? and content_hash in ({})
                    """.format(",".join("?" for _ in items_and_hashes)),
                    [collection_id, model_id]
                    + [item_and_hash[1] for item_and_hash in items_and_hashes],
                )
            ]
            filtered_batch = [item for item in batch if item[0] not in existing_ids]
            if not filtered_batch:
                continue
            embeddings = list(
                embedding_model.embed_multi(
                    (item[1] for item in filtered_batch), key=key
                )
            )
            # Replace the stale/missing rows in one atomic batch, so a crash
            # can never leave a row half-written and completed work persists
            with self.db.atomic():
                cast(Table, self.db["embeddings"]).insert_all(
                    (
                        {
                            "collection_id": collection_id,
                            "id": id,
                            "embedding": llm.encode(embedding),
                            "model": model_id,
                            "content": (
                                value if (store and isinstance(value, str)) else None
                            ),
                            "content_blob": (
                                value if (store and isinstance(value, bytes)) else None
                            ),
                            "content_hash": self.content_hash(value),
                            "metadata": json.dumps(metadata) if metadata else None,
                            "updated": int(time.time()),
                        }
                        for (embedding, (id, value, metadata)) in zip(
                            embeddings, filtered_batch
                        )
                    ),
                    replace=True,
                )

    def similar_by_vector(
        self,
        vector: list[float],
        number: int = 10,
        skip_id: str | None = None,
        prefix: str | None = None,
        *,
        model: str | None = None,
    ) -> list[Entry]:
        """
        Find similar items in the collection by a given vector.

        Only embeddings belonging to one model generation are searched, so
        vectors produced by incompatible models are never compared. By
        default that is the collection's configured model - meaning old
        results stay valid while a re-index to a new model is in progress.

        Args:
            vector (list): Vector to search by (must come from ``model``)
            number (int, optional): Number of similar items to return
            skip_id (str, optional): An ID to exclude from the results
            prefix: (str, optional): Filter results to IDs witih this prefix
            model (str, optional): Model generation to search; defaults to
                the collection's configured model

        Returns:
            list: List of Entry objects (each tagged with its ``.model``)
        """
        import llm

        scope_model = self.model_id if model is None else model

        def distance_score(other_encoded):
            other_vector = llm.decode(other_encoded)
            return llm.cosine_similarity(other_vector, vector)

        self.db.register_function(distance_score, replace=True)

        where_bits = ["collection_id = ?", "model = ?"]
        where_args: list[Any] = [str(self.id), scope_model]

        if prefix:
            where_bits.append("id LIKE ? || '%'")
            where_args.append(prefix)

        if skip_id:
            where_bits.append("id != ?")
            where_args.append(skip_id)

        return [
            Entry(
                id=row["id"],
                score=row["score"],
                content=row["content"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else None,
                model=row["model"],
            )
            for row in self.db.query(
                """
            select id, content, metadata, model, distance_score(embedding) as score
            from embeddings
            where {where}
            order by score desc limit {number}
        """.format(
                    where=" and ".join(where_bits),
                    number=number,
                ),
                where_args,
            )
        ]

    def similar_by_id(
        self, id: str, number: int = 10, prefix: str | None = None
    ) -> list[Entry]:
        """
        Find similar items in the collection by a given ID.

        The search is scoped to whatever model generation the pivot ID
        belongs to, so comparisons always stay within one vector space.

        Args:
            id (str): ID to search by
            number (int, optional): Number of similar items to return
            prefix: (str, optional): Filter results to IDs with this prefix

        Returns:
            list: List of Entry objects
        """
        import llm

        matches = list(
            self.db["embeddings"].rows_where(
                "collection_id = ? and id = ?", (self.id, id)
            )
        )
        if not matches:
            raise self.DoesNotExist("ID not found")
        pivot = matches[0]
        comparison_vector = llm.decode(pivot["embedding"])
        return self.similar_by_vector(
            comparison_vector,
            number,
            skip_id=id,
            prefix=prefix,
            model=pivot["model"],
        )

    def similar(
        self,
        value: str | bytes,
        number: int = 10,
        prefix: str | None = None,
        *,
        model: EmbeddingModel | str | None = None,
    ) -> list[Entry]:
        """
        Find similar items in the collection by a given value.

        Args:
            value (str or bytes): value to search by
            number (int, optional): Number of similar items to return
            prefix: (str, optional): Filter results to IDs with this prefix
            model (EmbeddingModel or str, optional): Model generation to
                embed the query with and search; defaults to the
                collection's configured model

        Returns:
            list: List of Entry objects
        """
        embedding_model = self._resolve_model(model)
        comparison_vector = embedding_model.embed(value)
        return self.similar_by_vector(
            comparison_vector, number, prefix=prefix, model=embedding_model.model_id
        )

    @classmethod
    def exists(cls, db: Database, name: str) -> bool:
        """
        Does this collection exist in the database?

        Args:
            name (str): Name of the collection
        """
        rows = list(db["collections"].rows_where("name = ?", [name]))
        return bool(rows)

    def delete(self):
        """
        Delete the collection and its embeddings from the database
        """
        with self.db.atomic():
            self.db.execute("delete from embeddings where collection_id = ?", [self.id])
            self.db.execute("delete from collections where id = ?", [self.id])

    def set_model(self, model: EmbeddingModel | str) -> None:
        """
        Switch the collection's configured model generation.

        This only updates which model the collection treats as current - it
        does not embed or delete anything. It is normally called once a
        re-index run has brought every row onto the new model.

        Args:
            model (EmbeddingModel or str): The new current model
        """
        embedding_model = self._resolve_model(model)
        if embedding_model.model_id != self.model_id:
            cast(Table, self.db["collections"]).update(
                self.id, {"model": embedding_model.model_id}
            )
            self.model_id = embedding_model.model_id
        self._model = embedding_model

    def stale_count(self, model: EmbeddingModel | str | None = None) -> int:
        """
        Count rows not yet embedded by the given model.

        A row counts as stale when its recorded model generation differs
        from ``model`` (which defaults to the collection's current model).
        Zero means the collection is consistent for that generation.

        Args:
            model (EmbeddingModel or str, optional): Target model; defaults
                to the collection's configured model

        Returns:
            int: Number of embeddings belonging to other model generations
        """
        target_model_id = self._resolve_model(model).model_id
        return next(
            self.db.query(
                """
            select count(*) as c from embeddings
            where collection_id = ? and coalesce(model, '') != ?
            """,
                (self.id, target_model_id),
            )
        )["c"]

    def model_counts(self) -> dict[str | None, int]:
        """
        Count how many embeddings the collection holds per model generation.

        Returns:
            dict: Mapping of model ID to number of embeddings
        """
        return {
            row["model"]: row["c"]
            for row in self.db.query(
                """
            select model, count(*) as c from embeddings
            where collection_id = ?
            group by model
            """,
                (self.id,),
            )
        }

    @staticmethod
    def content_hash(input: str | bytes) -> bytes:
        "Hash content for deduplication. Override to change hashing behavior."
        if isinstance(input, str):
            input = input.encode("utf8")
        return hashlib.md5(input).digest()
