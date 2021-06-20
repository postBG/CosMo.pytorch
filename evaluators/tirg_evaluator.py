from evaluators.abc import AbstractBaseEvaluator


class SimpleEvaluator(AbstractBaseEvaluator):
    def __init__(self, models, dataloaders, top_k=(1, 10, 50), visualizer=None):
        super().__init__(models, dataloaders, top_k, visualizer)
        self.lower_image_encoder = self.models['lower_image_encoder']
        self.upper_image_encoder = self.models['upper_image_encoder']
        self.text_encoder = self.models['text_encoder']
        self.compositor = self.models['layer4']

    def _extract_image_features(self, images):
        mid_features, _ = self.lower_image_encoder(images)
        return self.upper_image_encoder(mid_features)

    def _extract_original_and_composed_features(self, images, modifiers, len_modifiers):
        mid_image_features, _ = self.lower_image_encoder(images)
        text_features = self.text_encoder(modifiers, len_modifiers)
        composed_features, _ = self.compositor(mid_image_features, text_features)
        return self.upper_image_encoder(mid_image_features), self.upper_image_encoder(composed_features)
