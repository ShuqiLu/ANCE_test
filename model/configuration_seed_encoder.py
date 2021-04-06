class SEEDEncoderConfig(BertConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.RobertaModel` or a
    :class:`~transformers.TFRobertaModel`. It is used to instantiate a RoBERTa model according to the specified
    arguments, defining the model architecture.
    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.
    The :class:`~transformers.RobertaConfig` class directly inherits :class:`~transformers.BertConfig`. It reuses the
    same defaults. Please check the parent class for more information.
    Examples::
        >>> from transformers import RobertaConfig, RobertaModel
        >>> # Initializing a RoBERTa configuration
        >>> configuration = RobertaConfig()
        >>> # Initializing a model from the configuration
        >>> model = RobertaModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "seed_encoder"

    def __init__(self, pad_token_id=1, bos_token_id=0, eos_token_id=2, **kwargs):
        """Constructs RobertaConfig."""
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)