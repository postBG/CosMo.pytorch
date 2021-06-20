import wandb
from PIL import ImageOps, Image


def draw_border(img, color='red'):
    return ImageOps.expand(img, border=5, fill=color)


class RecallVisualizer(object):
    def __init__(self, test_dataloaders):
        self.test_dataset = test_dataloaders['samples'].dataset
        self.query_dataset = test_dataloaders['query'].dataset

    def __call__(self, sample_info, is_positive=True):
        prefix_label = "positive sample " if is_positive else "negative_sample"
        visualization_dict = {}
        for i, info in enumerate(sample_info):
            ref_idx = info['ref_idx']
            targ_idxs = info['targ_idxs']
            targ_scores = info['targ_scores']
            gt_score = info['gt_score']
            img_data = []
            ref_img, ref_gt, modifier, targ_gt, _ = self.query_dataset.__getitem__(ref_idx, use_transform=False)
            ref_caption = 'Ref: {}'.format(modifier)
            formatted_ref_img = self._crop_and_center_img(ref_img)
            img_data.append(wandb.Image(formatted_ref_img, caption=ref_caption))

            # Load GT
            gt_img = self.test_dataset.sample_img_for_visualizing(targ_gt)
            formatted_gt_img = self._crop_and_center_img(gt_img)
            img_data.append(wandb.Image(formatted_gt_img, caption='GT: {:.3f}'.format(gt_score)))

            for score, targ_idx in zip(targ_scores, targ_idxs):
                targ_img, targ_attr = self.test_dataset.__getitem__(targ_idx, use_transform=False)
                caption = targ_attr + ": {:.3f}".format(score)
                border_color = 'green' if targ_attr == targ_gt else 'red'
                formatted_targ_img = draw_border(targ_img, color=border_color)
                formatted_targ_img = self._crop_and_center_img(formatted_targ_img)
                img_data.append(wandb.Image(formatted_targ_img, caption=caption))
            visualization_dict[prefix_label + str(i)] = img_data
        return visualization_dict

    @staticmethod
    def _crop_and_center_img(img, background_size=(300, 500)):
        background_w, background_h = background_size
        background = Image.new('RGB', background_size, (255, 255, 255))
        img_w, img_h = img.size
        reduce_rate_w = background_w / img_w
        reduce_rate_h = background_h / background_w
        epsilon = 2
        if int(img_h * reduce_rate_w) <= background_h + epsilon:
            new_size = (int(img_w * reduce_rate_w), int(img_h * reduce_rate_w))
        else:
            new_size = (int(img_w * reduce_rate_h), int(img_h * reduce_rate_h))
        resized_img = img.resize(new_size)
        offset = ((background_w - new_size[0]) // 2, (background_h - new_size[1]) // 2)
        background.paste(resized_img, offset)
        return background
