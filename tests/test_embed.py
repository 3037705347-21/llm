import json
from unittest.mock import ANY

import pytest

import llm
from llm.embeddings import Entry
from llm.embeddings_migrations import embeddings_migrations


class NewGenerationModel(llm.EmbeddingModel):
    "Stand-in for a new embedding model generation (does not need registering)"

    model_id = "embed-v2"
    batch_size = 2

    def __init__(self):
        self.embedded = []
        self.batch_count = 0

    def embed_batch(self, texts):
        self.batch_count += 1
        for text in texts:
            self.embedded.append(text)
            yield [float(len(text))]


class FlakyNewGenerationModel(llm.EmbeddingModel):
    "Same model identity as NewGenerationModel, but can fail on chosen texts"

    model_id = "embed-v2"
    batch_size = 2

    def __init__(self, fail_on=()):
        self.fail_on = set(fail_on)
        self.embedded = []

    def embed_batch(self, texts):
        for text in texts:
            if text in self.fail_on:
                raise RuntimeError("temporary embedding API failure")
            self.embedded.append(text)
            yield [float(len(text))]


def test_demo_plugin():
    model = llm.get_embedding_model("embed-demo")
    assert model.embed("hello world") == [5, 5] + [0] * 14


def test_embed_key(embed_key_demo):
    assert embed_key_demo.embed("hello world", key="sekrit") == [5, 5] + [0] * 14
    assert embed_key_demo.keys == ["sekrit"]
    assert embed_key_demo.key is None


def test_embed_multi_key(embed_key_demo):
    results = list(
        embed_key_demo.embed_multi(
            ["hello world", "goodbye world"], batch_size=1, key="sekrit"
        )
    )
    assert results == [[5, 5] + [0] * 14, [7, 5] + [0] * 14]
    assert embed_key_demo.keys == ["sekrit", "sekrit"]
    assert embed_key_demo.key is None


def test_embed_key_legacy_plugin_fallback():
    class LegacyKeyEmbeddingModel(llm.EmbeddingModel):
        model_id = "legacy-key"
        needs_key = "legacy-key"

        def __init__(self):
            self.keys = []

        def embed_batch(self, items):
            self.keys.append(self.key)
            yield from ([float(len(item))] for item in items)

    model = LegacyKeyEmbeddingModel()
    assert model.embed("hello", key="sekrit") == [5.0]
    assert model.keys == ["sekrit"]
    assert model.key is None


@pytest.mark.parametrize(
    "batch_size,expected_batches",
    (
        (None, 100),
        (10, 100),
    ),
)
def test_embed_huge_list(batch_size, expected_batches):
    model = llm.get_embedding_model("embed-demo")
    huge_list = (f"hello {i}" for i in range(1000))
    kwargs = {}
    if batch_size:
        kwargs["batch_size"] = batch_size
    results = model.embed_multi(huge_list, **kwargs)
    assert repr(type(results)) == "<class 'generator'>"
    first_twos = {}
    for result in results:
        key = (result[0], result[1])
        first_twos[key] = first_twos.get(key, 0) + 1
    assert first_twos == {(5, 1): 10, (5, 2): 90, (5, 3): 900}
    assert model.batch_count == expected_batches


def test_embed_store(collection):
    collection.embed("3", "hello world again", store=True)
    assert collection.db["embeddings"].count == 3
    assert (
        next(collection.db["embeddings"].rows_where("id = ?", ["3"]))["content"]
        == "hello world again"
    )


def test_embed_metadata(collection):
    collection.embed("3", "hello yet again", metadata={"foo": "bar"}, store=True)
    assert collection.db["embeddings"].count == 3
    assert json.loads(
        next(collection.db["embeddings"].rows_where("id = ?", ["3"]))["metadata"]
    ) == {"foo": "bar"}
    entry = collection.similar("hello yet again")[0]
    assert entry.id == "3"
    assert entry.metadata == {"foo": "bar"}
    assert entry.content == "hello yet again"


def test_collection_embed_key(embed_key_demo, request):
    collection = llm.Collection("test", model=embed_key_demo)
    request.addfinalizer(collection.db.close)
    collection.embed("1", "hello world", key="sekrit")
    assert embed_key_demo.keys == ["sekrit"]
    assert embed_key_demo.key is None


def test_collection_embed_multi_key(embed_key_demo, request):
    collection = llm.Collection("test", model=embed_key_demo)
    request.addfinalizer(collection.db.close)
    collection.embed_multi([("1", "hello world"), ("2", "goodbye world")], key="sekrit")
    assert embed_key_demo.keys == ["sekrit"]
    assert embed_key_demo.key is None


def test_collection(collection):
    assert collection.id == 1
    assert collection.count() == 2
    # Check that the embeddings are there
    rows = list(collection.db["embeddings"].rows)
    assert rows == [
        {
            "collection_id": 1,
            "id": "1",
            "embedding": llm.encode([5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "model": "embed-demo",
            "content": None,
            "content_blob": None,
            "content_hash": collection.content_hash("hello world"),
            "metadata": None,
            "updated": ANY,
        },
        {
            "collection_id": 1,
            "id": "2",
            "embedding": llm.encode([7, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "model": "embed-demo",
            "content": None,
            "content_blob": None,
            "content_hash": collection.content_hash("goodbye world"),
            "metadata": None,
            "updated": ANY,
        },
    ]
    assert isinstance(rows[0]["updated"], int) and rows[0]["updated"] > 0


def test_similar(collection):
    results = list(collection.similar("hello world"))
    assert results == [
        Entry(
            id="1",
            score=pytest.approx(0.9999999999999999),
            model="embed-demo",
        ),
        Entry(
            id="2",
            score=pytest.approx(0.9863939238321437),
            model="embed-demo",
        ),
    ]


def test_similar_prefixed(collection):
    results = list(collection.similar("hello world", prefix="2"))
    assert results == [
        Entry(
            id="2",
            score=pytest.approx(0.9863939238321437),
            model="embed-demo",
        ),
    ]


def test_similar_by_id(collection):
    results = list(collection.similar_by_id("1"))
    assert results == [
        Entry(
            id="2",
            score=pytest.approx(0.9863939238321437),
            model="embed-demo",
        ),
    ]


@pytest.mark.parametrize(
    "batch_size,expected_batches",
    (
        (None, 100),
        (5, 200),
    ),
)
@pytest.mark.parametrize("with_metadata", (False, True))
def test_embed_multi(db_factory, with_metadata, batch_size, expected_batches):
    db = db_factory(memory=True)
    collection = llm.Collection("test", db, model_id="embed-demo")
    model = collection.model()
    assert getattr(model, "batch_count", 0) == 0
    ids_and_texts = ((str(i), f"hello {i}") for i in range(1000))
    kwargs = {}
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    if with_metadata:
        ids_and_texts = ((id, text, {"meta": id}) for id, text in ids_and_texts)
        collection.embed_multi_with_metadata(ids_and_texts, **kwargs)
    else:
        # Exercise store=True here too
        collection.embed_multi(ids_and_texts, store=True, **kwargs)
    rows = list(db["embeddings"].rows)
    assert len(rows) == 1000
    rows_with_metadata = [row for row in rows if row["metadata"] is not None]
    rows_with_content = [row for row in rows if row["content"] is not None]
    if with_metadata:
        assert len(rows_with_metadata) == 1000
        assert len(rows_with_content) == 0
    else:
        assert len(rows_with_metadata) == 0
        assert len(rows_with_content) == 1000
    # Every row should have content_hash set
    assert all(row["content_hash"] is not None for row in rows)
    # Check batch count
    assert collection.model().batch_count == expected_batches


def test_collection_delete(collection):
    db = collection.db
    assert db["embeddings"].count == 2
    assert db["collections"].count == 1
    collection.delete()
    assert db["embeddings"].count == 0
    assert db["collections"].count == 0


def test_binary_only_and_text_only_embedding_models():
    binary_only = llm.get_embedding_model("embed-binary-only")
    text_only = llm.get_embedding_model("embed-text-only")

    assert binary_only.supports_binary
    assert not binary_only.supports_text
    assert not text_only.supports_binary
    assert text_only.supports_text

    with pytest.raises(ValueError):
        binary_only.embed("hello world")

    binary_only.embed(b"hello world")

    with pytest.raises(ValueError):
        text_only.embed(b"hello world")

    text_only.embed("hello world")

    # Try the multi versions too
    # Have to call list() on this or the generator is not evaluated
    with pytest.raises(ValueError):
        list(binary_only.embed_multi(["hello world"]))

    list(binary_only.embed_multi([b"hello world"]))

    with pytest.raises(ValueError):
        list(text_only.embed_multi([b"hello world"]))

    list(text_only.embed_multi(["hello world"]))


def test_embeddings_migration_backfills_model(db_factory):
    # A database frozen at the pre-m006 schema
    db = db_factory(memory=True)
    embeddings_migrations.apply(db, stop_before="m006_add_embeddings_model")
    db.execute("insert into collections (id, name, model) values (1, 'old', 'old-gen')")
    db.execute(
        "insert into embeddings (collection_id, id, embedding) values (1, 'a', x'00')"
    )
    assert "model" not in db["embeddings"].columns_dict

    embeddings_migrations.apply(db)

    row = next(db["embeddings"].rows)
    # Existing rows inherit their collection's model generation
    assert row["model"] == "old-gen"
    # The foreign key survived the table rebuild
    assert any(
        fk.column == "collection_id"
        and fk.other_table == "collections"
        and fk.other_column == "id"
        for fk in db["embeddings"].foreign_keys
    )


def test_rows_tagged_with_model_generation(db_factory):
    db = db_factory(memory=True)
    collection = llm.Collection("test", db, model_id="embed-demo")
    collection.embed_multi([("1", "alpha"), ("2", "beta")])
    assert collection.model_counts() == {"embed-demo": 2}
    assert collection.stale_count() == 0
    assert {row["model"] for row in db["embeddings"].rows} == {"embed-demo"}


def test_embed_single_row_rotates_by_model_identity(db_factory):
    db = db_factory(memory=True)
    collection = llm.Collection("test", db, model_id="embed-demo")
    collection.embed("1", "hello world")

    # A different model generation means the row is stale and must be redone
    v2 = NewGenerationModel()
    collection.embed("1", "hello world", model=v2)
    assert v2.embedded == ["hello world"]
    assert collection.model_counts() == {"embed-v2": 1}
    # Replaced in place - no duplicate row
    assert db["embeddings"].count == 1

    # Same generation + unchanged content: not recomputed
    v2_again = NewGenerationModel()
    collection.embed("1", "hello world", model=v2_again)
    assert v2_again.embedded == []

    # Changed content with the same model: recomputed
    v2_changed = NewGenerationModel()
    collection.embed("1", "hello changed world", model=v2_changed)
    assert v2_changed.embedded == ["hello changed world"]


def test_embed_multi_rotation_only_replaces_stale_rows(db_factory):
    db = db_factory(memory=True)
    collection = llm.Collection("test", db, model_id="embed-demo")
    entries = [("1", "alpha"), ("2", "beta"), ("3", "gamma")]
    collection.embed_multi(entries, store=True)

    new_model = NewGenerationModel()
    collection.embed_multi(entries, model=new_model)

    # All rows replaced in place with the new generation
    assert db["embeddings"].count == 3
    assert collection.model_counts() == {"embed-v2": 3}
    assert sorted(row["id"] for row in db["embeddings"].rows) == ["1", "2", "3"]
    assert sorted(new_model.embedded) == ["alpha", "beta", "gamma"]
    assert collection.stale_count(new_model) == 0

    # Re-running with a fresh instance (a new process, same model identity)
    # recomputes nothing
    rerun_model = NewGenerationModel()
    collection.embed_multi(entries, model=rerun_model)
    assert rerun_model.embedded == []
    assert rerun_model.batch_count == 0
    assert collection.model_counts() == {"embed-v2": 3}


def test_embed_multi_rotation_fills_mixed_state(db_factory):
    db = db_factory(memory=True)
    collection = llm.Collection("test", db, model_id="embed-demo")
    collection.embed_multi([("1", "alpha"), ("2", "beta"), ("3", "gamma")], store=True)
    # Simulate an interrupted previous run: row 1 already migrated, row 3 gone
    collection.embed("1", "alpha", model=NewGenerationModel())
    db.execute("delete from embeddings where collection_id = ? and id = ?", [1, "3"])
    assert collection.model_counts() == {"embed-v2": 1, "embed-demo": 1}

    new_model = NewGenerationModel()
    collection.embed_multi(
        [("1", "alpha"), ("2", "beta"), ("3", "gamma")], model=new_model
    )

    # Only the stale row (2) and missing row (3) were computed
    assert sorted(new_model.embedded) == ["beta", "gamma"]
    assert db["embeddings"].count == 3
    assert collection.model_counts() == {"embed-v2": 3}


def test_embed_multi_rotation_resumes_after_batch_failure(db_factory):
    db = db_factory(memory=True)
    collection = llm.Collection("test", db, model_id="embed-demo")
    entries = [("1", "alpha"), ("2", "beta"), ("3", "gamma")]
    collection.embed_multi(entries, store=True)

    # First attempt dies while embedding the second batch (batch_size = 2)
    flaky = FlakyNewGenerationModel(fail_on={"gamma"})
    with pytest.raises(RuntimeError, match="temporary embedding API failure"):
        collection.embed_multi(entries, model=flaky, store=True)

    # Batch 1 was committed atomically; the failed batch left no half-row,
    # and the collection has NOT switched models
    assert db["embeddings"].count == 3
    assert collection.model_counts() == {"embed-v2": 2, "embed-demo": 1}
    assert collection.model_id == "embed-demo"

    # Old results remain usable throughout: default search stays in the old
    # generation, and every result says which generation it came from
    old_results = collection.similar("alpha")
    assert [(e.id, e.model) for e in old_results] == [("3", "embed-demo")]

    # New-generation rows are inspectable via an explicit generation scope
    new_results = collection.similar_by_vector([5.0], model="embed-v2")
    assert sorted(e.id for e in new_results) == ["1", "2"]
    assert {e.model for e in new_results} == {"embed-v2"}

    # Re-run with the same model identity now healthy: completed rows are
    # skipped and the run converges without duplicates or recomputation
    recovered = FlakyNewGenerationModel()
    collection.embed_multi(entries, model=recovered, store=True)
    assert recovered.embedded == ["gamma"]
    assert db["embeddings"].count == 3
    assert collection.model_counts() == {"embed-v2": 3}
    assert collection.stale_count(recovered) == 0

    collection.set_model(recovered)
    assert collection.model_id == "embed-v2"
    # Search defaults to the new generation once the switch is complete
    assert sorted(e.id for e in collection.similar("gamma")) == ["1", "2", "3"]


def test_similar_scoped_to_generation_during_rotation(db_factory):
    db = db_factory(memory=True)
    collection = llm.Collection("test", db, model_id="embed-demo")
    collection.embed_multi([("1", "alpha"), ("2", "beta"), ("3", "gamma")], store=True)
    v2 = NewGenerationModel()
    collection.embed_multi([("1", "alpha"), ("2", "beta")], model=v2)
    assert collection.model_id == "embed-demo"

    # While mixed, default search only sees the collection's generation
    results = collection.similar("alpha")
    assert [(e.id, e.model) for e in results] == [("3", "embed-demo")]

    # similar_by_id follows whatever generation the pivot row belongs to
    assert sorted(e.id for e in collection.similar_by_id("1")) == ["2"]
    assert all(e.model == "embed-v2" for e in collection.similar_by_id("1"))
    # Row 3 is alone in the old generation (and excludes itself)
    assert collection.similar_by_id("3") == []

    # Flipping with a row still stale keeps that row out of default search
    collection.set_model(v2)
    assert sorted(e.id for e in collection.similar("gamma")) == ["1", "2"]
    # Finish the last stale row and everything is consistent again
    collection.embed("3", "gamma", model=v2)
    assert sorted(e.id for e in collection.similar("gamma")) == ["1", "2", "3"]


def test_stale_count_and_set_model(db_factory):
    db = db_factory(memory=True)
    collection = llm.Collection("test", db, model_id="embed-demo")
    collection.embed_multi([("1", "alpha"), ("2", "beta")])
    v2 = NewGenerationModel()

    assert collection.stale_count(v2) == 2
    collection.set_model(v2)
    # set_model only flips the pointer; rows are untouched
    assert collection.model_id == "embed-v2"
    assert collection.model().model_id == "embed-v2"
    assert next(db["collections"].rows)["model"] == "embed-v2"
    assert collection.model_counts() == {"embed-demo": 2}
    assert collection.stale_count() == 2
    # Setting the same model again is a no-op
    collection.set_model(v2)
    assert collection.model_id == "embed-v2"
