trtllm-bench
===========================

trtllm-bench is a comprehensive benchmarking tool for TensorRT LLM engines. It provides three main subcommands for different benchmarking scenarios:

.. include:: ../_includes/note_sections.rst
   :start-after: .. start-note-config-flag-alias
   :end-before: .. end-note-config-flag-alias

Syntax
------

.. click:: tensorrt_llm.commands.bench:main
   :prog: trtllm-bench
   :nested: full
   :commands: throughput, latency, build



Dataset preparation
------------------

prepare-dataset
^^^^^^^^^^^^^^^

``trtllm-bench`` ships a built-in ``prepare-dataset`` subcommand for generating benchmark
datasets in the required JSONL format. Each line of the output file is a complete JSON entry
consumed by the ``throughput`` and ``latency`` subcommands.

The tokenizer is resolved from the ``--model`` (or ``--model_path``) argument passed to the
top-level ``trtllm-bench`` command.

**Usage:**

.. code-block:: bash

    trtllm-bench --model <model-id> prepare-dataset [OPTIONS] SUBCOMMAND [SUBCOMMAND-OPTIONS]

**Options**

----

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Option
     - Description
   * - ``--output``
     - Output JSONL filename (default: ``preprocessed_dataset.json``)
   * - ``--random-seed``
     - Random seed for reproducible token generation (default: 420)
   * - ``--task-id``
     - Fixed LoRA task ID assigned to all requests (default: -1, disabled)
   * - ``--rand-task-id <min> <max>``
     - Randomly assign a LoRA task ID in ``[min, max]`` per request
   * - ``--lora-dir``
     - Parent directory containing LoRA adapter subdirectories named by task ID
   * - ``--trust-remote-code``
     - Trust remote code when loading the tokenizer (env: ``TRUST_REMOTE_CODE``)
   * - ``--log-level``
     - Logging level: ``info`` or ``debug`` (default: ``info``)

**Subcommands:**

real-dataset
""""""""""""

Build a dataset from a real HuggingFace dataset. Supports three input modes detected
automatically from the data shape:

- **Single-turn text** — a string field, optionally prefixed by a prompt.
- **Multi-turn conversation** — a list of strings per row (e.g. MT-Bench ``turns`` field).
- **Multimodal** — rows containing ``image`` / ``image_1`` keys (images only; video not yet supported).

.. code-block:: bash

    trtllm-bench --model meta-llama/Llama-3.1-8B \
      prepare-dataset --output /tmp/data.jsonl \
      real-dataset \
      --dataset-name cnn_dailymail \
      --dataset-config-name 3.0.0 \
      --dataset-split test \
      --dataset-input-key article \
      --dataset-output-key highlights \
      --num-requests 500

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Option
     - Description
   * - ``--dataset-name``
     - HuggingFace dataset name (required)
   * - ``--dataset-config-name``
     - Dataset config name, if the dataset has multiple configs
   * - ``--dataset-split``
     - Dataset split to use, e.g. ``train``, ``validation``, ``test`` (required)
   * - ``--dataset-input-key``
     - Dictionary key for the input text field
   * - ``--dataset-prompt-key``
     - Dictionary key for a per-row prompt prefix
   * - ``--dataset-prompt``
     - Literal prompt string to prepend to every input (alternative to ``--dataset-prompt-key``)
   * - ``--dataset-output-key``
     - Dictionary key for the golden output; its tokenized length sets ``output_tokens``
   * - ``--dataset-image-key``
     - Dictionary key for images (multimodal; default: ``image``)
   * - ``--num-requests``
     - Maximum number of requests; capped to the dataset size
   * - ``--max-input-len``
     - Filter out requests whose tokenized input exceeds this length
   * - ``--output-len-dist``
     - Override output lengths with a normal distribution: ``<mean>,<stdev>`` (required for multimodal and multi-turn)


token-norm-dist
"""""""""""""""

Generate a synthetic dataset whose input and output sequence lengths are drawn from
normal distributions.

.. code-block:: bash

    trtllm-bench --model meta-llama/Llama-3.1-8B \
      prepare-dataset --output /tmp/data.txt \
      token-norm-dist \
      --num-requests 1000 \
      --input-mean 128 --input-stdev 0 \
      --output-mean 128 --output-stdev 0

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Option
     - Description
   * - ``--num-requests``
     - Number of requests to generate (required)
   * - ``--input-mean``
     - Normal distribution mean for input sequence length in tokens (required)
   * - ``--input-stdev``
     - Normal distribution standard deviation for input length (required; set to 0 for fixed length)
   * - ``--output-mean``
     - Normal distribution mean for output sequence length in tokens (required)
   * - ``--output-stdev``
     - Normal distribution standard deviation for output length (required; set to 0 for fixed length)


token-unif-dist
"""""""""""""""

Generate a synthetic dataset whose input and output sequence lengths are drawn from
uniform distributions.

.. code-block:: bash

    trtllm-bench --model meta-llama/Llama-3.1-8B \
      prepare-dataset --output /tmp/data.txt \
      token-unif-dist \
      --num-requests 1000 \
      --input-min 64 --input-max 256 \
      --output-min 64 --output-max 256

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Option
     - Description
   * - ``--num-requests``
     - Number of requests to generate (required)
   * - ``--input-min``
     - Inclusive lower bound for input sequence length (required)
   * - ``--input-max``
     - Inclusive upper bound for input sequence length (required)
   * - ``--output-min``
     - Inclusive lower bound for output sequence length (required)
   * - ``--output-max``
     - Inclusive upper bound for output sequence length (required)
