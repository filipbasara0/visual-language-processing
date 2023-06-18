import torch.nn as nn

from model.feature_extractor import ResNetFeatureExtractor
from model.pos_embs import SinePositionalEncoding
from model.transformer import get_transformer


def model_config_factory(model_name):
    transformers_dict = {
        "encoder_decoder_sm": {
            "model_dim": 256,
            "ff_dim": 1024,
            "num_heads": 8,
            "num_layers": 6,
            "feature_map_size": 2048,
            "dec_div": 1
        },
        "encoder_decoder_base": {
            "model_dim": 512,
            "ff_dim": 2048,
            "num_heads": 16,
            "num_layers": 12,
            "feature_map_size": 2048,
            "dec_div": 1
        },
        "encoder_decoder_lg": {
            "model_dim": 768,
            "ff_dim": 4096,
            "num_heads": 16,
            "num_layers": 12,
            "feature_map_size": 2048,
            "dec_div": 2
        },
        "encoder_decoder_xlg": {
            "model_dim": 768,
            "ff_dim": 4096,
            "num_heads": 24,
            "num_layers": 24,
            "feature_map_size": 2048,
            "dec_div": 4
        }
    }

    return transformers_dict.get(model_name,
                                 transformers_dict["encoder_decoder_base"])


class VLP(nn.Module):

    def __init__(self, model_dim, num_layers, ff_dim, num_heads,
                 feature_map_size, vocab_size, dropout, transformer_type,
                 dec_div):
        super(VLP, self).__init__()

        self.feature_extractor = nn.Sequential(
            ResNetFeatureExtractor(feature_map_size=feature_map_size,
                                   out_features_size=model_dim),
            SinePositionalEncoding(model_dim))

        self.transformer = get_transformer(num_layers=num_layers,
                                           model_dim=model_dim,
                                           ff_dim=ff_dim,
                                           num_heads=num_heads,
                                           vocab_size=vocab_size,
                                           dropout=dropout,
                                           transformer_type=transformer_type,
                                           dec_div=dec_div)

    def get_image_features(self, images):
        return self.feature_extractor(images)

    def forward(self, images, tgt=None, tgt_mask=None):
        image_features = self.get_image_features(images)
        return self.transformer(image_features, tgt, tgt_mask=tgt_mask)


class VLPForTextRecognition(nn.Module):

    def __init__(self,
                 model_dim,
                 num_layers,
                 num_heads,
                 ff_dim,
                 feature_map_size,
                 vocab_size,
                 dec_div=2,
                 dropout=0.0):
        super(VLPForTextRecognition, self).__init__()

        self.vlp = VLP(num_layers=num_layers,
                       model_dim=model_dim,
                       ff_dim=ff_dim,
                       num_heads=num_heads,
                       feature_map_size=feature_map_size,
                       vocab_size=vocab_size,
                       dropout=dropout,
                       transformer_type="encoder_decoder",
                       dec_div=dec_div)
        self.cls = nn.Linear(model_dim, vocab_size)

    def forward(self, images, tgt, tgt_mask=None):
        out = self.vlp(images, tgt, tgt_mask)
        return self.cls(out)


class VLPForTokenClassification(nn.Module):

    def __init__(self,
                 model_dim,
                 num_layers,
                 num_heads,
                 ff_dim,
                 feature_map_size,
                 vocab_size,
                 num_ner_tags,
                 num_pos_tags,
                 dec_div=2,
                 dropout=0.0):
        super(VLPForTokenClassification, self).__init__()

        self.vlp = VLP(num_layers=num_layers,
                       model_dim=model_dim,
                       ff_dim=ff_dim,
                       num_heads=num_heads,
                       feature_map_size=feature_map_size,
                       vocab_size=vocab_size,
                       dropout=dropout,
                       transformer_type="encoder_decoder",
                       dec_div=dec_div)
        self.cls = nn.Linear(model_dim, vocab_size)
        self.ner_cls = nn.Linear(model_dim, num_ner_tags)
        self.pos_cls = nn.Linear(model_dim, num_pos_tags)

    def forward(self, images, tgt, tgt_mask=None):
        out = self.vlp(images, tgt, tgt_mask)
        return self.cls(out), self.ner_cls(out), self.pos_cls(out)


class VLPForTextClassification(nn.Module):

    def __init__(self,
                 model_dim,
                 num_layers,
                 num_heads,
                 ff_dim,
                 feature_map_size,
                 num_classes,
                 dec_div=2,
                 dropout=0.0):
        super(VLPForTextClassification, self).__init__()

        self.vlp = VLP(num_layers=num_layers,
                       model_dim=model_dim,
                       ff_dim=ff_dim,
                       num_heads=num_heads,
                       feature_map_size=feature_map_size,
                       vocab_size=None,
                       dropout=dropout,
                       transformer_type="encoder",
                       dec_div=dec_div)
        self.out = nn.Linear(model_dim, num_classes)

    def forward(self, images):
        out = self.vlp(images)
        out = out.mean(dim=1)
        return self.out(out)


class VLPForTextMLM(nn.Module):

    def __init__(self,
                 model_dim,
                 num_layers,
                 num_heads,
                 ff_dim,
                 feature_map_size,
                 num_classes,
                 dec_div=2,
                 dropout=0.0):
        super(VLPForTextMLM, self).__init__()

        self.vlp = VLP(num_layers=num_layers,
                       model_dim=model_dim,
                       ff_dim=ff_dim,
                       num_heads=num_heads,
                       feature_map_size=feature_map_size,
                       vocab_size=None,
                       dropout=dropout,
                       transformer_type="encoder",
                       dec_div=dec_div)
        self.out = nn.Linear(model_dim, num_classes)

    def forward(self, images):
        out = self.vlp(images)
        return self.out(out)