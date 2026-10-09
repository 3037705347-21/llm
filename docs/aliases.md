(aliases)=
# Model aliases

LLM supports model aliases, which allow you to refer to a model by a short name instead of its full ID.

## Listing aliases

To list current aliases, run this:

```bash
llm aliases
```
Example output:

<!-- [[[cog
from click.testing import CliRunner
from llm.cli import cli
result = CliRunner().invoke(cli, ["aliases", "list"])
cog.out("```\n{}```".format(result.output))
]]] -->
```
4o                  : gpt-4o
4o-mini             : gpt-4o-mini
4.1                 : gpt-4.1
4.1-mini            : gpt-4.1-mini
4.1-nano            : gpt-4.1-nano
3.5                 : gpt-3.5-turbo
chatgpt             : gpt-3.5-turbo
chatgpt-16k         : gpt-3.5-turbo-16k
3.5-16k             : gpt-3.5-turbo-16k
4                   : gpt-4
gpt4                : gpt-4
gpt-4-turbo-preview : gpt-4-turbo
4-turbo             : gpt-4-turbo
4t                  : gpt-4-turbo
3.5-instruct        : gpt-3.5-turbo-instruct
chatgpt-instruct    : gpt-3.5-turbo-instruct
ada                 : text-embedding-ada-002 (embedding)
ada-002             : text-embedding-ada-002 (embedding)
3-small             : text-embedding-3-small (embedding)
3-large             : text-embedding-3-large (embedding)
3-small-512         : text-embedding-3-small-512 (embedding)
3-large-256         : text-embedding-3-large-256 (embedding)
3-large-1024        : text-embedding-3-large-1024 (embedding)
```
<!-- [[[end]]] -->

Add `--json` to get that list back as JSON:

```bash
llm aliases list --json
```
Example output:
```json
{
    "3.5": "gpt-3.5-turbo",
    "chatgpt": "gpt-3.5-turbo",
    "4": "gpt-4",
    "gpt4": "gpt-4",
    "ada": "ada-002"
}
```

## Adding a new alias

The `llm aliases set <alias> <model-id>` command can be used to add a new alias:

```bash
llm aliases set luna gpt-5.6-luna
```
You can also pass one or more `-q search` options to set an alias on the first model matching those search terms:
```bash
llm aliases set luna -q gpt -q luna
```
Now you can run the `gpt-5.6-luna` model using the `luna` alias like this:
```bash
llm -m luna 'An epic Greek-style saga about a cheesecake that builds a SQL database from scratch'
```
Aliases can be set for both regular models and {ref}`embedding models <embeddings>` using the same command. To set an alias of `oai` for the OpenAI `ada-002` embedding model use this:
```bash
llm aliases set oai ada-002
```
Now you can embed a string using that model like so:
```bash
llm embed -c 'hello world' -m oai
```
Output:
```
[-0.014945968054234982, 0.0014304015785455704, ...]
```

## Removing an alias

The `llm aliases remove <alias>` command will remove the specified alias:

```bash
llm aliases remove mini
```

## Switching an alias to another model

Your logs record the model (which may be an alias) that produced each response, alongside a `resolved_model` column holding the concrete model name when it is known. If you repoint an existing alias at a new model, older log entries that just name the alias would otherwise become ambiguous.

To prevent that, every time `llm aliases set` changes an alias from one model to another, the switch is appended to an `alias_rotations.json` file in the LLM configuration directory. The historical logs are then backfilled automatically: entries that used the alias and have no resolved model yet are updated to the concrete model the alias pointed at when they were logged.

For example, switching the `luna` alias:

```bash
llm aliases set luna gpt-6.0-luna
```
will record that `luna` previously pointed at `gpt-5.6-luna` and resolve the older log entries accordingly. Entries that already carry a resolved model name are never modified.

## Viewing alias rotations

To see the recorded rotations for all aliases:

```bash
llm aliases history
```
Pass an alias name to show just that alias:
```bash
llm aliases history luna
```

## Re-running the backfill

The backfill is idempotent: running it again only fills in entries that are still missing a resolved model, so it is safe to run as often as you like. If a backfill is interrupted partway through, re-running it picks up where it left off and converges on the same final state.

```bash
llm aliases backfill
```
The command reports how many log records it resolved. It needs both `aliases.json` and `alias_rotations.json` to be readable; if either file cannot be parsed it exits without modifying the database, so an unreadable file can never leave a half-backfilled log.

## Viewing the aliases file

Aliases are stored in an `aliases.json` file in the LLM configuration directory.

To see the path to that file, run this:

```bash
llm aliases path
```
To view the content of that file, run this:

```bash
cat "$(llm aliases path)"
```
