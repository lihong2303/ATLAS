
import torch
from torch import nn
from typing import Optional,Union,Tuple
from transformers.modeling_utils import PreTrainedModel
from transformers.models.vilt.configuration_vilt import ViltConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.utils.doc import add_start_docstrings,add_start_docstrings_to_model_forward,replace_return_docstrings
from Vilt.vilt import ViltEncoder,ViltEmbeddings,ViltPooler
from definitions import Adapter


_CONFIG_FOR_DOC = "ViltConfig"

class ViltPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ViltConfig
    base_model_prefix = "vilt"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ViltSelfAttention"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, ViltEncoder):
            module.gradient_checkpointing = value


VILT_START_DOCSTRING = r"""
    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ subclass. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ViltConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

VILT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary. Indices can be obtained using [`BertTokenizer`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details. [What are input
            IDs?](../glossary#input-ids)

        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)

        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:
            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.
            [What are token type IDs?](../glossary#token-type-ids)

        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`ViltImageProcessor.__call__`] for details.

        pixel_mask (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:

            - 1 for pixels that are real (i.e. **not masked**),
            - 0 for pixels that are padding (i.e. **masked**).
            `What are attention masks? <../glossary.html#attention-mask>`__

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.

        image_embeds (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`, *optional*):
            Optionally, instead of passing `pixel_values`, you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `pixel_values` into patch embeddings.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

VILT_IMAGES_AND_TEXT_CLASSIFICATION_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary. Indices can be obtained using [`BertTokenizer`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details. [What are input
            IDs?](../glossary#input-ids)

        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)

        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:
            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.
            [What are token type IDs?](../glossary#token-type-ids)

        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_images, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`ViltImageProcessor.__call__`] for details.

        pixel_mask (`torch.LongTensor` of shape `(batch_size, num_images, height, width)`, *optional*):
            Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:

            - 1 for pixels that are real (i.e. **not masked**),
            - 0 for pixels that are padding (i.e. **masked**).
            `What are attention masks? <../glossary.html#attention-mask>`__

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.

        image_embeds (`torch.FloatTensor` of shape `(batch_size, num_images, num_patches, hidden_size)`, *optional*):
            Optionally, instead of passing `pixel_values`, you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `pixel_values` into patch embeddings.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare ViLT Model transformer outputting raw hidden-states without any specific head on top.",
    VILT_START_DOCSTRING,
)



class ViltModel(ViltPreTrainedModel):
    def __init__(self, 
                 config, 
                 add_pooling_layer=True,
                 target_model:str=None,
                 adapter_weighted_method:str=None,
                 continual_sequence:str = None,
                 cur_dataset:str="",
                 update_method:str=None,
                 adapter:Optional[Adapter] = None
                 ):
        super().__init__(config)
        self.config = config

        self.embeddings = ViltEmbeddings(config)
        self.encoder = ViltEncoder(config,
                                   target_model=target_model,
                                   adapter_weighted_method=adapter_weighted_method,
                                   continual_sequence = continual_sequence,
                                   cur_dataset=cur_dataset,
                                   update_method=update_method,
                                   adapter=adapter)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViltPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.text_embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.text_embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(VILT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        image_token_type_idx: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[BaseModelOutputWithPooling, Tuple[torch.FloatTensor]]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import ViltProcessor, ViltModel
        >>> from PIL import Image
        >>> import requests

        >>> # prepare image and text
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "hello world"

        >>> processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        >>> model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")

        >>> inputs = processor(image, text, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        text_batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((text_batch_size, seq_length)), device=device)

        if pixel_values is not None and image_embeds is not None:
            raise ValueError("You cannot specify both pixel_values and image_embeds at the same time")
        elif pixel_values is None and image_embeds is None:
            raise ValueError("You have to specify either pixel_values or image_embeds")

        image_batch_size = pixel_values.shape[0] if pixel_values is not None else image_embeds.shape[0]
        if image_batch_size != text_batch_size:
            raise ValueError("The text inputs and image inputs need to have the same batch size")
        if pixel_mask is None:
            pixel_mask = torch.ones((image_batch_size, self.config.image_size, self.config.image_size), device=device)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output, attention_mask = self.embeddings(
            input_ids,
            attention_mask,
            token_type_ids,
            pixel_values,
            pixel_mask,
            inputs_embeds,
            image_embeds,
            image_token_type_idx=image_token_type_idx,
        )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )