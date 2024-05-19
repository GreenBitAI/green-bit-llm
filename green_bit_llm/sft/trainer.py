import json
import shutil
import os
from typing import Optional

from trl import SFTTrainer

from green_bit_llm.common.utils import STRATEGY_FILE_NAME
from green_bit_llm.common.utils import get_model_path


class GbaSFTTrainer(SFTTrainer):
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Saves the model to the specified directory and also ensures that the
        'quant_strategy.json' file is copied over to the same directory.

        Args:
            output_dir (Optional[str]): The directory to which the model and the
                                        'quant_strategy.json' file should be saved.
                                        If None, the model will be saved to the default location.
            _internal_call (bool): A flag used to indicate whether this method was
                                   called internally by the library, which can affect
                                   certain behaviors (not used in this override).

        Raises:
            ValueError: If the expected GBA prefix is not found in the output directory path.
        """

        # Perform the original save model behavior of the superclass
        # out_dir should be os.path.join(args.save_dir, args.model)
        super().save_model(output_dir)

        # Define the prefix to look for in the output directory path
        gba_prefix = "GreenBitAI" + os.path.sep
        # Find the prefix in the output directory path
        start_index = output_dir.find(gba_prefix)

        if start_index == -1:
            config_path = os.path.join(output_dir, "config.json")
            if os.path.isfile(config_path):
                with open(config_path, 'r') as file:
                    data = json.load(file)
                if "quantization_config" in data.keys():
                    quantization_config = data["quantization_config"]
                    if "exllama_config" in quantization_config.keys():
                        del quantization_config["exllama_config"]
                    if "use_exllama" in quantization_config.keys():
                        del quantization_config["use_exllama"]

                with open(config_path, 'w') as file:
                    json.dump(data, file, indent=4)
            return

        # Ensure this is executed only on the main process
        if not self.is_world_process_zero():
            return

        # save "quant_strategy.json" file
        start_pos = start_index + len(gba_prefix) - 1
        end_pos = output_dir.find(os.path.sep, start_pos + 1)

        if end_pos == -1:
            model_name = output_dir[start_index:]
        else:
            model_name = output_dir[start_index:end_pos]

        model_from_path = get_model_path(model_name)
        quant_strategy_file = os.path.join(model_from_path, STRATEGY_FILE_NAME)
        destination_path = os.path.join(output_dir, STRATEGY_FILE_NAME)
        shutil.copy(quant_strategy_file, destination_path)
