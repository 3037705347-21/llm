class ModelError(Exception):
    "Models can raise this error, which will be displayed to the user"


class NeedsKeyException(ModelError):
    "Model needs an API key which has not been provided"


class ConversationNotSupported(ValueError):
    "A single-turn model was given assistant or tool results"


class AliasHistoryError(ValueError):
    "The aliases file or alias rotation history cannot be read or parsed"
