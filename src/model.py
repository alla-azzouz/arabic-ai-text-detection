# src/model.py

import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoConfig, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput


class AdvancedHybridModel(PreTrainedModel):
    """
    Hybrid classifier:
    - Transformer backbone for text representation
    - Explicit feature tower for stylometric / engineered features
    - Joint classifier head
    """
    config_class = AutoConfig
    base_model_prefix = "transformer"
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.num_explicit_features = getattr(config, "num_explicit_features", 0)

        # Build transformer architecture first.
        # Pretrained weights will be loaded manually in load_advanced_model().
        self.transformer = AutoModel.from_config(config)

        # Explicit feature tower
        feature_tower_hidden_dim = 64
        self.feature_tower = nn.Sequential(
            nn.LayerNorm(self.num_explicit_features),
            nn.Linear(self.num_explicit_features, feature_tower_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # Final classifier
        combined_input_size = config.hidden_size + feature_tower_hidden_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(combined_input_size),
            nn.Linear(combined_input_size, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.num_labels),
        )

        # Initialize newly added layers
        self.post_init()

    @property
    def embeddings(self):
        return self.transformer.embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        explicit_features=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if explicit_features is None:
            raise ValueError("explicit_features must be provided.")

        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
        )

        # Mean pooling over valid tokens
        last_hidden_state = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        text_embedding = sum_embeddings / sum_mask

        # Feature branch
        feature_embedding = self.feature_tower(explicit_features)

        # Concatenate text + explicit features
        combined_embedding = torch.cat((text_embedding, feature_embedding), dim=1)
        logits = self.classifier(combined_embedding)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def load_advanced_model(model_name_or_path, num_explicit_features, **kwargs):
    """
    Robust loading strategy:
    1. Build the custom wrapper from config
    2. Explicitly load pretrained backbone weights into loaded_model.transformer
    3. Keep feature tower and classifier randomly initialized for downstream training
    """
    print(f"Loading model: {model_name_or_path} | Features: {num_explicit_features}")

    config = AutoConfig.from_pretrained(model_name_or_path, **kwargs)
    config._attn_implementation = "eager"
    config.num_explicit_features = num_explicit_features

    # Step 1: build custom wrapper
    loaded_model = AdvancedHybridModel(config)

    # Step 2: explicitly load pretrained transformer backbone
    print("Loading pretrained transformer weights manually...")
    pretrained_transformer = AutoModel.from_pretrained(
        model_name_or_path,
        config=config,
    )

    missing, unexpected = loaded_model.transformer.load_state_dict(
        pretrained_transformer.state_dict(),
        strict=False,
    )

    del pretrained_transformer

    print(f"Transformer missing keys   : {len(missing)}")
    print(f"Transformer unexpected keys: {len(unexpected)}")

    if len(missing) > 0:
        print("Sample missing keys:", missing[:10])
    if len(unexpected) > 0:
        print("Sample unexpected keys:", unexpected[:10])

    
    sample_param = next(loaded_model.transformer.parameters())
    std = sample_param.data.float().std().item()
    mean = sample_param.data.float().mean().item()
    print(f"Transformer encoder std={std:.6f}, mean={mean:.6f}")
    print("Pretrained transformer weights loaded manually ✅")
    print("Model ready.")

    return loaded_model