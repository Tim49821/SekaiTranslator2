# Adapted from https://github.com/jingsongliujing/OnnxOCR
import math
import re

import cv2
import numpy as np
import onnxruntime


class BaseRecLabelDecode:
    def __init__(self, character_dict_path=None, use_space_char=False):
        self.reverse = False
        self.character_str = []

        if character_dict_path is None:
            self.character_str = list("0123456789abcdefghijklmnopqrstuvwxyz")
        else:
            with open(character_dict_path, "rb") as fin:
                for line in fin.readlines():
                    self.character_str.append(line.decode("utf-8").strip("\n").strip("\r\n"))
            if use_space_char:
                self.character_str.append(" ")
            if "arabic" in character_dict_path:
                self.reverse = True

        self.character = self.add_special_char(list(self.character_str))
        self.dict = {char: idx for idx, char in enumerate(self.character)}

    def pred_reverse(self, pred):
        pred_re = []
        current = ""
        for char in pred:
            if not bool(re.search("[a-zA-Z0-9 :*./%+-]", char)):
                if current:
                    pred_re.append(current)
                pred_re.append(char)
                current = ""
            else:
                current += char
        if current:
            pred_re.append(current)
        return "".join(pred_re[::-1])

    def add_special_char(self, dict_character):
        return dict_character

    def get_ignored_tokens(self):
        return [0]

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [
                self.character[text_id]
                for text_id in text_index[batch_idx][selection]
            ]
            conf_list = text_prob[batch_idx][selection] if text_prob is not None else [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]

            text = "".join(char_list)
            if self.reverse:
                text = self.pred_reverse(text)
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list


class CTCLabelDecode(BaseRecLabelDecode):
    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, (tuple, list)):
            preds = preds[-1]
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        if label is None:
            return text
        return text, self.decode(label)

    def add_special_char(self, dict_character):
        return ["blank"] + dict_character


class PredictBase:
    def get_onnx_session(self, model_dir, device):
        if device == "cuda":
            providers = [
                ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "EXHAUSTIVE"}),
                "CPUExecutionProvider",
            ]
        elif device == "mps":
            providers = [
                ("CoreMLExecutionProvider", {"MLComputeUnits": "ALL"}),
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]
        return onnxruntime.InferenceSession(model_dir, None, providers=providers)

    def get_output_name(self, onnx_session):
        return [node.name for node in onnx_session.get_outputs()]

    def get_input_name(self, onnx_session):
        return [node.name for node in onnx_session.get_inputs()]

    def get_input_feed(self, input_name, image_numpy):
        return {name: image_numpy for name in input_name}


class TextRecognizer(PredictBase):
    def __init__(self, args):
        self.rec_image_shape = [int(v) for v in args.rec_image_shape.split(",")]
        self.rec_batch_num = args.rec_batch_num
        self.rec_algorithm = args.rec_algorithm
        self.postprocess_op = CTCLabelDecode(
            character_dict_path=args.rec_char_dict_path,
            use_space_char=args.use_space_char,
        )
        self.rec_onnx_session = self.get_onnx_session(args.rec_model_dir, args.device)
        self.rec_input_name = self.get_input_name(self.rec_onnx_session)
        self.rec_output_name = self.get_output_name(self.rec_onnx_session)

    def resize_norm_img(self, img, max_wh_ratio):
        img_c, img_h, img_w = self.rec_image_shape
        assert img_c == img.shape[2]
        img_w = int(img_h * max_wh_ratio)

        h, w = img.shape[:2]
        ratio = w / float(h)
        resized_w = img_w if math.ceil(img_h * ratio) > img_w else int(math.ceil(img_h * ratio))
        resized_image = cv2.resize(img, (resized_w, img_h))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5

        padding_im = np.zeros((img_c, img_h, img_w), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def __call__(self, img_list):
        img_num = len(img_list)
        width_list = [img.shape[1] / float(img.shape[0]) for img in img_list]
        indices = np.argsort(np.array(width_list))
        rec_res = [["", 0.0]] * img_num
        batch_num = self.rec_batch_num

        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            _, img_h, img_w = self.rec_image_shape[:3]
            max_wh_ratio = img_w / img_h
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                max_wh_ratio = max(max_wh_ratio, w * 1.0 / h)
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]], max_wh_ratio)
                norm_img_batch.append(norm_img[np.newaxis, :])

            norm_img_batch = np.concatenate(norm_img_batch).copy()
            input_feed = self.get_input_feed(self.rec_input_name, norm_img_batch)
            outputs = self.rec_onnx_session.run(self.rec_output_name, input_feed=input_feed)
            rec_result = self.postprocess_op(outputs[0])
            for rno, result in enumerate(rec_result):
                rec_res[indices[beg_img_no + rno]] = result

        return rec_res
