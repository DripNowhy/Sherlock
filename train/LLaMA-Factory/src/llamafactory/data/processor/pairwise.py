# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, infer_seqlen


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)

REFINE_TEMPLATE = """<image>Below is a QUESTION from a user and an EXAMPLE RESPONSE.
Please provide a more helpful RESPONSE, improving the EXAMPLE RESPONSE by making the content even clearer, more accurate, and with a reasonable logic.
Focus on addressing the human's QUESTION step by step based on the image without including irrelevant content.

QUESTION:
{Question}

EXAMPLE RESPONSE:
{Example_Response}

Now, refine and improve the RESPONSE further. You can consider two approaches:
1. REFINEMENT: If the SUMMARY section in the response is closely related to the question, the CAPTION section accurately describes the image, the REASONING section is logically clear and correct without any contradictions, and the CONCLUSION provides an accurate answer based on the previous steps, enhance clarity, accuracy, or reasoning logic as needed.
2. NEW RESPONSE: If the SUMMARY section incorrectly summarizes the intent of the issue, the CAPTION contains content unrelated to or incorrect about the image, there are logical errors or contradictions in the REASONING, or the CONCLUSION incorrectly states the findings, please enhance the accuracy and quality of each step, and craft a more effective RESPONSE that thoroughly resolves the QUESTION.

RESPONSE:
"""

class PairwiseDatasetProcessor(DatasetProcessor):
    def _encode_data_example(
        self,
        prompt: Sequence[Dict[str, str]],
        response: Sequence[Dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        audios: Sequence["AudioInput"],
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        chosen_messages = self.template.mm_plugin.process_messages(
            prompt + [response[0]], images, videos, audios, self.processor
        )
        rejected_messages = self.template.mm_plugin.process_messages(
            prompt + [response[1]], images, videos, audios, self.processor
        )
        prompt_ids, chosen_ids = self.template.encode_oneturn(self.tokenizer, chosen_messages, system, tools)
        _, rejected_ids = self.template.encode_oneturn(self.tokenizer, rejected_messages, system, tools)

        if self.template.efficient_eos:
            chosen_ids += [self.tokenizer.eos_token_id]
            rejected_ids += [self.tokenizer.eos_token_id]

        prompt_ids, _ = self.template.mm_plugin.process_token_ids(
            prompt_ids, None, images, videos, audios, self.tokenizer, self.processor
        )
        # consider the response is more important
        source_len, target_len = infer_seqlen(
            len(prompt_ids), max(len(chosen_ids), len(rejected_ids)), self.data_args.cutoff_len
        )
        prompt_ids = prompt_ids[:source_len]
        chosen_ids = chosen_ids[:target_len]
        rejected_ids = rejected_ids[:target_len]

        chosen_input_ids = prompt_ids + chosen_ids
        chosen_labels = [IGNORE_INDEX] * source_len + chosen_ids
        rejected_input_ids = prompt_ids + rejected_ids
        rejected_labels = [IGNORE_INDEX] * source_len + rejected_ids
        return chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels

    def preprocess_dataset(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) < 2:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            model_inputs["chosen_input_ids"].append(chosen_input_ids)
            model_inputs["chosen_attention_mask"].append([1] * len(chosen_input_ids))
            model_inputs["chosen_labels"].append(chosen_labels)
            model_inputs["rejected_input_ids"].append(rejected_input_ids)
            model_inputs["rejected_attention_mask"].append([1] * len(rejected_input_ids))
            model_inputs["rejected_labels"].append(rejected_labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        return model_inputs

    def print_data_example(self, example: Dict[str, List[int]]) -> None:
        valid_chosen_labels = list(filter(lambda x: x != IGNORE_INDEX, example["chosen_labels"]))
        valid_rejected_labels = list(filter(lambda x: x != IGNORE_INDEX, example["rejected_labels"]))
        print("chosen_input_ids:\n{}".format(example["chosen_input_ids"]))
        print(
            "chosen_inputs:\n{}".format(self.tokenizer.decode(example["chosen_input_ids"], skip_special_tokens=False))
        )
        print("chosen_label_ids:\n{}".format(example["chosen_labels"]))
        print(f"chosen_labels:\n{self.tokenizer.decode(valid_chosen_labels, skip_special_tokens=False)}")
        print("rejected_input_ids:\n{}".format(example["rejected_input_ids"]))
        print(
            "rejected_inputs:\n{}".format(
                self.tokenizer.decode(example["rejected_input_ids"], skip_special_tokens=False)
            )
        )
        print("rejected_label_ids:\n{}".format(example["rejected_labels"]))
        print(f"rejected_labels:\n{self.tokenizer.decode(valid_rejected_labels, skip_special_tokens=False)}")

class RefineDatasetProcessor(DatasetProcessor):
    def _encode_data_example(
        self,
        prompt: Sequence[Dict[str, str]],
        response: Sequence[Dict[str, str]],
        prefix: Optional[str],
        prefix_l: Optional[str],
        system: Optional[str],
        tools: Optional[str],
        weights: Sequence[int],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        audios: Sequence["AudioInput"],
    ) -> Tuple[List[int], List[int], List[int], List[int]]:

        llamav_input_template = (
            '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question}'
            '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{prefix}'
        )
        llamav_response_template = (
            '{suffix}<|eot_id|>'
        )

        chosen_messages = self.template.mm_plugin.process_messages(
            prompt + [response[0]], images, videos, audios, self.processor
        )
        question = prompt[0]["content"]
        chosen_res = response[0]["content"]
        rejected_res = response[1]["content"]
        chosen_refined_w_messages = self.template.mm_plugin.process_messages(
            [{"content": REFINE_TEMPLATE.format(Question=question.replace('<image>', ''), Example_Response=prefix_l+rejected_res), "role": "user"}] + [response[0]], images, videos, audios, self.processor
        )
        rejected_refined_w_messages = self.template.mm_plugin.process_messages(
            [{"content": REFINE_TEMPLATE.format(Question=question.replace('<image>', ''), Example_Response=prefix+chosen_res), "role": "user"}] + [response[0]], images, videos, audios, self.processor
        )
        chosen_correction_w_messages = self.template.mm_plugin.process_messages(
            [{"content": REFINE_TEMPLATE.format(Question=question.replace('<image>', ''), Example_Response=prefix_l+rejected_res), "role": "user"}] + [response[0]], images, videos, audios, self.processor
        )
        rejected_correction_w_messages = self.template.mm_plugin.process_messages(
            [{"content": REFINE_TEMPLATE.format(Question=question.replace('<image>', ''), Example_Response=prefix+chosen_res), "role": "user"}] + [response[0]], images, videos, audios, self.processor
        )

        prompt_w_text = llamav_input_template.format(question=chosen_messages[0]['content'], prefix="")
        prompt_l_text = llamav_input_template.format(question=chosen_messages[0]['content'], prefix="")
        chosen_text = llamav_response_template.format(suffix=prefix+chosen_res)
        rejected_text = llamav_response_template.format(suffix=prefix_l+rejected_res)

        chosen_refined_w_prompt_text = llamav_input_template.format(question=chosen_refined_w_messages[0]['content'], prefix=prefix)
        chosen_refined_w_text = llamav_response_template.format(suffix=chosen_res)
        chosen_refined_l_prompt_text = llamav_input_template.format(question=chosen_refined_w_messages[0]['content'], prefix=prefix_l)
        chosen_refined_l_text = llamav_response_template.format(suffix=rejected_res)

        rejected_refined_w_prompt_text = llamav_input_template.format(question=rejected_refined_w_messages[0]['content'], prefix=prefix)
        rejected_refined_w_text = llamav_response_template.format(suffix=chosen_res)
        rejected_refined_l_prompt_text = llamav_input_template.format(question=rejected_refined_w_messages[0]['content'], prefix=prefix_l)
        rejected_refined_l_text = llamav_response_template.format(suffix=rejected_res)

        chosen_correction_w_prompt_text = llamav_input_template.format(question=chosen_correction_w_messages[0]['content'], prefix=prefix_l)
        chosen_correction_w_text = llamav_response_template.format(suffix=chosen_res)
        chosen_correction_l_prompt_text = llamav_input_template.format(question=chosen_correction_w_messages[0]['content'], prefix=prefix)
        chosen_correction_l_text = llamav_response_template.format(suffix=rejected_res)

        rejected_correction_w_prompt_text = llamav_input_template.format(question=rejected_correction_w_messages[0]['content'], prefix=prefix_l)
        rejected_correction_w_text = llamav_response_template.format(suffix=chosen_res)
        rejected_correction_l_prompt_text = llamav_input_template.format(question=rejected_correction_w_messages[0]['content'], prefix=prefix)
        rejected_correction_l_text = llamav_response_template.format(suffix=rejected_res)

        prompt_w_ids = self.tokenizer.encode(prompt_w_text, add_special_tokens=False)
        chosen_ids = self.tokenizer.encode(chosen_text, add_special_tokens=False)
        prompt_l_ids = self.tokenizer.encode(prompt_l_text, add_special_tokens=False)
        rejected_ids = self.tokenizer.encode(rejected_text, add_special_tokens=False)
        chosen_refined_w_prompt_ids = self.tokenizer.encode(chosen_refined_w_prompt_text, add_special_tokens=False)
        chosen_refined_w_ids = self.tokenizer.encode(chosen_refined_w_text, add_special_tokens=False)
        chosen_refined_l_prompt_ids = self.tokenizer.encode(chosen_refined_l_prompt_text, add_special_tokens=False)
        chosen_refined_l_ids = self.tokenizer.encode(chosen_refined_l_text, add_special_tokens=False)
        rejected_refined_w_prompt_ids = self.tokenizer.encode(rejected_refined_w_prompt_text, add_special_tokens=False)
        rejected_refined_w_ids = self.tokenizer.encode(rejected_refined_w_text, add_special_tokens=False)
        rejected_refined_l_prompt_ids = self.tokenizer.encode(rejected_refined_l_prompt_text, add_special_tokens=False)
        rejected_refined_l_ids = self.tokenizer.encode(rejected_refined_l_text, add_special_tokens=False)
        chosen_correction_w_prompt_ids = self.tokenizer.encode(chosen_correction_w_prompt_text, add_special_tokens=False)
        chosen_correction_w_ids = self.tokenizer.encode(chosen_correction_w_text
        , add_special_tokens=False)
        chosen_correction_l_prompt_ids = self.tokenizer.encode(chosen_correction_l_prompt_text, add_special_tokens=False)
        chosen_correction_l_ids = self.tokenizer.encode(chosen_correction_l_text, add_special_tokens=False)
        rejected_correction_w_prompt_ids = self.tokenizer.encode(rejected_correction_w_prompt_text, add_special_tokens=False)
        rejected_correction_w_ids = self.tokenizer.encode(rejected_correction_w_text, add_special_tokens=False)
        rejected_correction_l_prompt_ids = self.tokenizer.encode(rejected_correction_l_prompt_text, add_special_tokens=False)
        rejected_correction_l_ids = self.tokenizer.encode(rejected_correction_l_text, add_special_tokens=False)

        if self.template.efficient_eos:
            chosen_ids += [self.tokenizer.eos_token_id]
            chosen_refined_w_ids += [self.tokenizer.eos_token_id]
            chosen_refined_l_ids += [self.tokenizer.eos_token_id]
            rejected_ids += [self.tokenizer.eos_token_id]
            rejected_refined_w_ids += [self.tokenizer.eos_token_id]
            rejected_refined_l_ids += [self.tokenizer.eos_token_id]
            chosen_correction_w_ids += [self.tokenizer.eos_token_id]
            chosen_correction_l_ids += [self.tokenizer.eos_token_id]
            rejected_correction_w_ids += [self.tokenizer.eos_token_id]
            rejected_correction_l_ids += [self.tokenizer.eos_token_id]

        prompt_w_ids, _ = self.template.mm_plugin.process_token_ids(
            prompt_w_ids, None, images, videos, audios, self.tokenizer, self.processor
        )
        prompt_l_ids, _ = self.template.mm_plugin.process_token_ids(
            prompt_l_ids, None, images, videos, audios, self.tokenizer, self.processor
        )
        chosen_refined_w_prompt_ids, _ = self.template.mm_plugin.process_token_ids(
            chosen_refined_w_prompt_ids, None, images, videos, audios, self.tokenizer, self.processor
        )
        chosen_refined_l_prompt_ids, _ = self.template.mm_plugin.process_token_ids(
            chosen_refined_l_prompt_ids, None, images, videos, audios, self.tokenizer, self.processor
        )
        rejected_refined_w_prompt_ids, _ = self.template.mm_plugin.process_token_ids(
            rejected_refined_w_prompt_ids, None, images, videos, audios, self.tokenizer, self.processor
        )
        rejected_refined_l_prompt_ids, _ = self.template.mm_plugin.process_token_ids(
            rejected_refined_l_prompt_ids, None, images, videos, audios, self.tokenizer, self.processor
        )
        chosen_correction_w_prompt_ids, _ = self.template.mm_plugin.process_token_ids(
            chosen_correction_w_prompt_ids, None, images, videos, audios, self.tokenizer, self.processor
        )
        chosen_correction_l_prompt_ids, _ = self.template.mm_plugin.process_token_ids(
            chosen_correction_l_prompt_ids, None, images, videos, audios, self.tokenizer, self.processor
        )
        rejected_correction_w_prompt_ids, _ = self.template.mm_plugin.process_token_ids(
            rejected_correction_w_prompt_ids, None, images, videos, audios, self.tokenizer, self.processor
        )
        rejected_correction_l_prompt_ids, _ = self.template.mm_plugin.process_token_ids(
            rejected_correction_l_prompt_ids, None, images, videos, audios, self.tokenizer, self.processor
        )
        # consider the response is more important
        source_w_len, target_w_len = infer_seqlen(
            len(prompt_w_ids), len(chosen_ids), self.data_args.cutoff_len
        )
        source_l_len, target_l_len = infer_seqlen(
            len(prompt_l_ids), len(chosen_ids), self.data_args.cutoff_len
        )
        source_chosen_refined_w_len, target_chosen_refined_w_len = infer_seqlen(
            len(chosen_refined_w_prompt_ids), len(chosen_refined_w_ids), self.data_args.cutoff_len
        )
        source_chosen_refined_l_len, target_chosen_refined_l_len = infer_seqlen(
            len(chosen_refined_l_prompt_ids), len(chosen_refined_l_ids), self.data_args.cutoff_len
        )
        source_rejected_refined_w_len, target_rejected_refined_w_len = infer_seqlen(
            len(rejected_refined_w_prompt_ids), len(rejected_refined_w_ids), self.data_args.cutoff_len
        )
        source_rejected_refined_l_len, target_rejected_refined_l_len = infer_seqlen(
            len(rejected_refined_l_prompt_ids), len(rejected_refined_l_ids), self.data_args.cutoff_len
        )
        source_chosen_correction_w_len, target_chosen_correction_w_len = infer_seqlen(
            len(chosen_correction_w_prompt_ids), len(chosen_correction_w_ids), self.data_args.cutoff_len
        )
        source_chosen_correction_l_len, target_chosen_correction_l_len = infer_seqlen(
            len(chosen_correction_l_prompt_ids), len(chosen_correction_l_ids), self.data_args.cutoff_len
        )
        source_rejected_correction_w_len, target_rejected_correction_w_len = infer_seqlen(
            len(rejected_correction_w_prompt_ids), len(rejected_correction_w_ids), self.data_args.cutoff_len
        )
        source_rejected_correction_l_len, target_rejected_correction_l_len = infer_seqlen(
            len(rejected_correction_l_prompt_ids), len(rejected_correction_l_ids), self.data_args.cutoff_len
        )

        prompt_w_ids = prompt_w_ids[:source_w_len]
        prompt_l_ids = prompt_l_ids[:source_l_len]
        chosen_ids = chosen_ids[:target_w_len]
        chosen_refined_w_prompt_ids = chosen_refined_w_prompt_ids[:source_chosen_refined_w_len]
        chosen_refined_w_ids = chosen_refined_w_ids[:target_chosen_refined_w_len]
        chosen_refined_l_prompt_ids = chosen_refined_l_prompt_ids[:source_chosen_refined_l_len]
        chosen_refined_l_ids = chosen_refined_l_ids[:target_chosen_refined_l_len]
        rejected_ids = rejected_ids[:target_l_len]
        rejected_refined_w_prompt_ids = rejected_refined_w_prompt_ids[:source_rejected_refined_w_len]
        rejected_refined_w_ids = rejected_refined_w_ids[:target_rejected_refined_w_len]
        rejected_refined_l_prompt_ids = rejected_refined_l_prompt_ids[:source_rejected_refined_l_len]
        rejected_refined_l_ids = rejected_refined_l_ids[:target_rejected_refined_l_len]
        chosen_correction_w_prompt_ids = chosen_correction_w_prompt_ids[:source_chosen_correction_w_len]
        chosen_correction_w_ids = chosen_correction_w_ids[:target_chosen_correction_w_len]
        chosen_correction_l_prompt_ids = chosen_correction_l_prompt_ids[:source_chosen_correction_l_len]
        chosen_correction_l_ids = chosen_correction_l_ids[:target_chosen_correction_l_len]
        rejected_correction_w_prompt_ids = rejected_correction_w_prompt_ids[:source_rejected_correction_w_len]
        rejected_correction_w_ids = rejected_correction_w_ids[:target_rejected_correction_w_len]
        rejected_correction_l_prompt_ids = rejected_correction_l_prompt_ids[:source_rejected_correction_l_len]
        rejected_correction_l_ids = rejected_correction_l_ids[:target_rejected_correction_l_len]

        chosen_input_ids = prompt_w_ids + chosen_ids
        chosen_labels = [IGNORE_INDEX] * source_w_len + chosen_ids
        chosen_refined_w_input_ids = chosen_refined_w_prompt_ids + chosen_refined_w_ids
        chosen_refined_w_labels = [IGNORE_INDEX] * (source_chosen_refined_w_len) + chosen_refined_w_ids
        chosen_refined_l_input_ids = chosen_refined_l_prompt_ids + chosen_refined_l_ids
        chosen_refined_l_labels = [IGNORE_INDEX] * (source_chosen_refined_l_len)+ chosen_refined_l_ids
        rejected_input_ids = prompt_l_ids + rejected_ids
        rejected_labels = [IGNORE_INDEX] * source_l_len + rejected_ids
        rejected_refined_w_input_ids = rejected_refined_w_prompt_ids + rejected_refined_w_ids
        rejected_refined_w_labels = [IGNORE_INDEX] * (source_rejected_refined_w_len) + rejected_refined_w_ids
        rejected_refined_l_input_ids = rejected_refined_l_prompt_ids + rejected_refined_l_ids
        rejected_refined_l_labels = [IGNORE_INDEX] * (source_rejected_refined_l_len) + rejected_refined_l_ids
        chosen_correction_w_input_ids = chosen_correction_w_prompt_ids + chosen_correction_w_ids
        chosen_correction_w_labels = [IGNORE_INDEX] * (source_chosen_correction_w_len) + chosen_correction_w_ids
        chosen_correction_l_input_ids = chosen_correction_l_prompt_ids + chosen_correction_l_ids
        chosen_correction_l_labels = [IGNORE_INDEX] * (source_chosen_correction_l_len) + chosen_correction_l_ids
        rejected_correction_w_input_ids = rejected_correction_w_prompt_ids + rejected_correction_w_ids
        rejected_correction_w_labels = [IGNORE_INDEX] * (source_rejected_correction_w_len) + rejected_correction_w_ids
        rejected_correction_l_input_ids = rejected_correction_l_prompt_ids + rejected_correction_l_ids
        rejected_correction_l_labels = [IGNORE_INDEX] * (source_rejected_correction_l_len) + rejected_correction_l_ids
        return chosen_input_ids, chosen_labels, chosen_refined_w_input_ids, chosen_refined_w_labels, chosen_refined_l_input_ids, chosen_refined_l_labels, rejected_input_ids, rejected_labels, rejected_refined_w_input_ids, rejected_refined_w_labels, rejected_refined_l_input_ids, rejected_refined_l_labels, chosen_correction_w_input_ids, chosen_correction_w_labels, chosen_correction_l_input_ids, chosen_correction_l_labels, rejected_correction_w_input_ids, rejected_correction_w_labels, rejected_correction_l_input_ids, rejected_correction_l_labels
    def preprocess_dataset(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) < 2:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            chosen_input_ids, chosen_labels, chosen_refined_w_input_ids, chosen_refined_w_labels, chosen_refined_l_input_ids, chosen_refined_l_labels, rejected_input_ids, rejected_labels, rejected_refined_w_input_ids, rejected_refined_w_labels, rejected_refined_l_input_ids, rejected_refined_l_labels, chosen_correction_w_input_ids, chosen_correction_w_labels, chosen_correction_l_input_ids, chosen_correction_l_labels, rejected_correction_w_input_ids, rejected_correction_w_labels, rejected_correction_l_input_ids, rejected_correction_l_labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                prefix=examples["_prefix"][i],
                prefix_l=examples["_prefix_l"][i],
                weights=examples["_weights"][i] or [],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            model_inputs["chosen_input_ids"].append(chosen_input_ids)
            model_inputs["chosen_attention_mask"].append([1] * len(chosen_input_ids))
            model_inputs["chosen_labels"].append(chosen_labels)
            model_inputs["rejected_input_ids"].append(rejected_input_ids)
            model_inputs["rejected_attention_mask"].append([1] * len(rejected_input_ids))
            model_inputs["rejected_labels"].append(rejected_labels)
            model_inputs["chosen_refined_w_input_ids"].append(chosen_refined_w_input_ids)
            model_inputs["chosen_refined_w_attention_mask"].append([1] * len(chosen_refined_w_input_ids))
            model_inputs["chosen_refined_w_labels"].append(chosen_refined_w_labels)
            model_inputs["chosen_refined_l_input_ids"].append(chosen_refined_l_input_ids)
            model_inputs["chosen_refined_l_attention_mask"].append([1] * len(chosen_refined_l_input_ids))
            model_inputs["chosen_refined_l_labels"].append(chosen_refined_l_labels)
            model_inputs["rejected_refined_w_input_ids"].append(rejected_refined_w_input_ids)
            model_inputs["rejected_refined_w_attention_mask"].append([1] * len(rejected_refined_w_input_ids))
            model_inputs["rejected_refined_w_labels"].append(rejected_refined_w_labels)
            model_inputs["rejected_refined_l_input_ids"].append(rejected_refined_l_input_ids)
            model_inputs["rejected_refined_l_attention_mask"].append([1] * len(rejected_refined_l_input_ids))
            model_inputs["rejected_refined_l_labels"].append(rejected_refined_l_labels)
            model_inputs["chosen_correction_w_input_ids"].append(chosen_correction_w_input_ids)
            model_inputs["chosen_correction_w_attention_mask"].append([1] * len(chosen_correction_w_input_ids))
            model_inputs["chosen_correction_w_labels"].append(chosen_correction_w_labels)
            model_inputs["chosen_correction_l_input_ids"].append(chosen_correction_l_input_ids)
            model_inputs["chosen_correction_l_attention_mask"].append([1] * len(chosen_correction_l_input_ids))
            model_inputs["chosen_correction_l_labels"].append(chosen_correction_l_labels)
            model_inputs["rejected_correction_w_input_ids"].append(rejected_correction_w_input_ids)
            model_inputs["rejected_correction_w_attention_mask"].append([1] * len(rejected_correction_w_input_ids))
            model_inputs["rejected_correction_w_labels"].append(rejected_correction_w_labels)
            model_inputs["rejected_correction_l_input_ids"].append(rejected_correction_l_input_ids)
            model_inputs["rejected_correction_l_attention_mask"].append([1] * len(rejected_correction_l_input_ids))
            model_inputs["rejected_correction_l_labels"].append(rejected_correction_l_labels)
            model_inputs["weights"].append(examples["_weights"][i])
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        return model_inputs

    def print_data_example(self, example: Dict[str, List[int]]) -> None:
        valid_chosen_labels = list(filter(lambda x: x != IGNORE_INDEX, example["chosen_labels"]))
        valid_chosen_refined_w_labels = list(filter(lambda x: x != IGNORE_INDEX, example["chosen_refined_w_labels"]))
        valid_chosen_refined_l_labels = list(filter(lambda x: x != IGNORE_INDEX, example["chosen_refined_l_labels"]))
        valid_rejected_labels = list(filter(lambda x: x != IGNORE_INDEX, example["rejected_labels"]))
        valid_rejected_refined_w_labels = list(filter(lambda x: x != IGNORE_INDEX, example["rejected_refined_w_labels"]))
        valid_rejected_refined_l_labels = list(filter(lambda x: x != IGNORE_INDEX, example["rejected_refined_l_labels"]))
        print("chosen_input_ids:\n{}".format(example["chosen_input_ids"]))
        print(
            "chosen_inputs:\n{}".format(self.tokenizer.decode(example["chosen_input_ids"], skip_special_tokens=False))
        )
        print("chosen_label_ids:\n{}".format(example["chosen_labels"]))
        print(f"chosen_labels:\n{self.tokenizer.decode(valid_chosen_labels, skip_special_tokens=False)}")

        print("chosen_refined_w_input_ids:\n{}".format(example["chosen_refined_w_input_ids"]))
        print(
            "chosen_refined_w_inputs:\n{}".format(
                self.tokenizer.decode(example["chosen_refined_w_input_ids"], skip_special_tokens=False)
            )
        )
        print("chosen_refined_w_label_ids:\n{}".format(example["chosen_refined_w_labels"]))
        print(f"chosen_refined_w_labels:\n{self.tokenizer.decode(valid_chosen_refined_w_labels, skip_special_tokens=False)}")

        print("chosen_refined_l_input_ids:\n{}".format(example["chosen_refined_l_input_ids"]))
        print(
            "chosen_refined_l_inputs:\n{}".format(
                self.tokenizer.decode(example["chosen_refined_l_input_ids"], skip_special_tokens=False)
            )
        )
        print("chosen_refined_l_label_ids:\n{}".format(example["chosen_refined_l_labels"]))
        print(f"chosen_refined_l_labels:\n{self.tokenizer.decode(valid_chosen_refined_l_labels, skip_special_tokens=False)}")

        print("chosen_correction_w_input_ids:\n{}".format(example["chosen_correction_w_input_ids"]))
        print(
            "chosen_correction_w_inputs:\n{}".format(
                self.tokenizer.decode(example["chosen_correction_w_input_ids"], skip_special_tokens=False)
            )
        )
        print("chosen_correction_w_label_ids:\n{}".format(example["chosen_correction_w_labels"]))
        print(f"chosen_correction_w_labels:\n{self.tokenizer.decode(valid_chosen_refined_l_labels, skip_special_tokens=False)}")

        print("chosen_correction_l_input_ids:\n{}".format(example["chosen_correction_l_input_ids"]))
        print(
            "chosen_correction_l_inputs:\n{}".format(
                self.tokenizer.decode(example["chosen_correction_l_input_ids"], skip_special_tokens=False)
            )
        )
        print("chosen_correction_l_label_ids:\n{}".format(example["chosen_correction_l_labels"]))
        print(f"chosen_correction_l_labels:\n{self.tokenizer.decode(valid_chosen_refined_l_labels, skip_special_tokens=False)}")

        print("rejected_input_ids:\n{}".format(example["rejected_input_ids"]))
        print(
            "rejected_inputs:\n{}".format(
                self.tokenizer.decode(example["rejected_input_ids"], skip_special_tokens=False)
            )
        )
        print("rejected_label_ids:\n{}".format(example["rejected_labels"]))
        print(f"rejected_labels:\n{self.tokenizer.decode(valid_rejected_labels, skip_special_tokens=False)}")

        print("rejected_refined_w_input_ids:\n{}".format(example["rejected_refined_w_input_ids"]))
        print(
            "rejected_refined_w_inputs:\n{}".format(
                self.tokenizer.decode(example["rejected_refined_w_input_ids"], skip_special_tokens=False)
            )
        )
        print("rejected_refined_w_label_ids:\n{}".format(example["rejected_refined_w_labels"]))
        print(f"rejected_refined_w_labels:\n{self.tokenizer.decode(valid_rejected_refined_w_labels, skip_special_tokens=False)}")

        print("rejected_refined_l_input_ids:\n{}".format(example["rejected_refined_l_input_ids"]))
        print(
            "rejected_refined_l_inputs:\n{}".format(
                self.tokenizer.decode(example["rejected_refined_l_input_ids"], skip_special_tokens=False)
            )
        )
        print("rejected_refined_l_label_ids:\n{}".format(example["rejected_refined_l_labels"]))
        print(f"rejected_refined_l_labels:\n{self.tokenizer.decode(valid_rejected_refined_l_labels, skip_special_tokens=False)}")

        print("rejected_correction_w_input_ids:\n{}".format(example["rejected_correction_w_input_ids"]))
        print(
            "rejected_correction_w_inputs:\n{}".format(
                self.tokenizer.decode(example["rejected_correction_w_input_ids"], skip_special_tokens=False)
            )
        )
        print("rejected_correction_w_label_ids:\n{}".format(example["rejected_correction_w_labels"]))
        print(f"rejected_correction_w_labels:\n{self.tokenizer.decode(valid_rejected_refined_l_labels, skip_special_tokens=False)}")

        print("rejected_correction_l_input_ids:\n{}".format(example["rejected_correction_l_input_ids"]))
        print(
            "rejected_correction_l_inputs:\n{}".format(
                self.tokenizer.decode(example["rejected_correction_l_input_ids"], skip_special_tokens=False)
            )
        )
        print("rejected_correction_l_label_ids:\n{}".format(example["rejected_correction_l_labels"]))
        print(f"rejected_correction_l_labels:\n{self.tokenizer.decode(valid_rejected_refined_l_labels, skip_special_tokens=False)}")

        print("weights:\n{}".format(example["weights"]))
