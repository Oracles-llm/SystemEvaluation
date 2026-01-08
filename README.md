system_evaluation/
├── .env                  # API keys (Create this yourself)
├── dataset_downloader.py # Your script to get data
├── run_eval.py           # Main execution script
├── src/
│   ├── __init__.py
│   ├── configurations.py
│   └── my_system.py      # YOUR system logic goes here
└── evaluation/
    ├── __init__.py
    ├── judges.py         # The AI Judge logic
    ├── utils.py          # Standard data structures
    └── layers/
        ├── __init__.py
        ├── layer1_accuracy.py
        ├── layer2_instruction.py
        ├── layer3_rag.py
        └── layer4_efficiency.py