# Description: This script runs the experiments that generates the results for the "Ablation Study on Dowmstream Tasks" in the paper, that is, Figure 7

# The following set of commands uses vampnet-base, that is, the base vampnet model.

python3 train_downstream.py -c Unit --backbone vampnet-base --seed 0

python3 train_downstream.py -c Unit --backbone vampnet-base --seed 1

python3 train_downstream.py -c Unit --backbone vampnet-base --seed 2

python3 train_downstream.py -c ELKI2name --top_classes 5 --backbone vampnet-base --grad-accum-steps 2 --seed 0

python3 train_downstream.py -c ELKI2name --top_classes 5 --backbone vampnet-base --grad-accum-steps 2 --seed 1

python3 train_downstream.py -c ELKI2name --top_classes 5 --backbone vampnet-base --grad-accum-steps 2 --seed 2

python3 train_downstream.py -c handv --backbone vampnet-base --seed 0

python3 train_downstream.py -c handv --backbone vampnet-base --seed 1

python3 train_downstream.py -c handv --backbone vampnet-base --seed 2

python3 train_downstream.py --binary -c ELKI2name --backbone vampnet-base --csv_path data/annotations/allcodas_with_noise.csv  --seed 0

python3 train_downstream.py --binary -c ELKI2name --backbone vampnet-base --csv_path data/annotations/allcodas_with_noise.csv  --seed 1

python3 train_downstream.py --binary -c ELKI2name --backbone vampnet-base --csv_path data/annotations/allcodas_with_noise.csv  --seed 2


# The following set of commands uses a finetuned VampNet model that has not gone through domain adaptation.

python3 train_downstream.py -c Unit --backbone vampnet-latest --seed 0

python3 train_downstream.py -c Unit --backbone vampnet-latest --seed 1

python3 train_downstream.py -c Unit --backbone vampnet-latest --seed 2

python3 train_downstream.py -c ELKI2name --top_classes 5 --backbone vampnet-latest --grad-accum-steps 2 --seed 0

python3 train_downstream.py -c ELKI2name --top_classes 5 --backbone vampnet-latest --grad-accum-steps 2 --seed 1

python3 train_downstream.py -c ELKI2name --top_classes 5 --backbone vampnet-latest --grad-accum-steps 2 --seed 2

python3 train_downstream.py -c handv --backbone vampnet-latest --seed 0

python3 train_downstream.py -c handv --backbone vampnet-latest --seed 1

python3 train_downstream.py -c handv --backbone vampnet-latest --seed 2

python3 train_downstream.py --binary -c ELKI2name --backbone vampnet-latest --csv_path data/annotations/allcodas_with_noise.csv  --seed 0

python3 train_downstream.py --binary -c ELKI2name --backbone vampnet-latest --csv_path data/annotations/allcodas_with_noise.csv  --seed 1

python3 train_downstream.py --binary -c ELKI2name --backbone vampnet-latest --csv_path data/annotations/allcodas_with_noise.csv  --seed 2

# The following set of commands uses only the tokenizer of Vampnet

python3 train_downstream.py --codec_only -c Unit --backbone vampnet-base --seed 0

python3 train_downstream.py --codec_only -c Unit --backbone vampnet-base --seed 1

python3 train_downstream.py --codec_only -c Unit --backbone vampnet-base --seed 2

python3 train_downstream.py --codec_only -c ELKI2name --top_classes 5 --backbone vampnet-base --grad-accum-steps 2 --seed 0

python3 train_downstream.py --codec_only -c ELKI2name --top_classes 5 --backbone vampnet-base --grad-accum-steps 2 --seed 1

python3 train_downstream.py --codec_only -c ELKI2name --top_classes 5 --backbone vampnet-base --grad-accum-steps 2 --seed 2

python3 train_downstream.py --codec_only -c handv --backbone vampnet-base --seed 0

python3 train_downstream.py --codec_only -c handv --backbone vampnet-base --seed 1

python3 train_downstream.py --codec_only -c handv --backbone vampnet-base --seed 2

python3 train_downstream.py --codec_only --binary -c ELKI2name --backbone vampnet-base --csv_path data/annotations/allcodas_with_noise.csv  --seed 0

python3 train_downstream.py --codec_only --binary -c ELKI2name --backbone vampnet-base --csv_path data/annotations/allcodas_with_noise.csv  --seed 1

python3 train_downstream.py --codec_only --binary -c ELKI2name --backbone vampnet-base --csv_path data/annotations/allcodas_with_noise.csv  --seed 2