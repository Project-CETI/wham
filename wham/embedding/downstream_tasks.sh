# This script contains all the experiment scripts needed to reproduce Table 1 in the WhAM paper.

# Coda Detection Task (~22 min for each vampnet experiment)

python3 train_downstream.py --binary -c ELKI2name --backbone aves --csv_path data/annotations/allcodas_with_noise.csv --random_init

python3 train_downstream.py --binary -c ELKI2name --backbone aves --csv_path data/annotations/allcodas_with_noise.csv  --seed 0

python3 train_downstream.py --binary -c ELKI2name --backbone aves --csv_path data/annotations/allcodas_with_noise.csv  --seed 1

python3 train_downstream.py --binary -c ELKI2name --backbone aves --csv_path data/annotations/allcodas_with_noise.csv  --seed 2

python3 train_downstream.py --binary -c ELKI2name --backbone vampnet-best --csv_path data/annotations/allcodas_with_noise.csv  --seed 0

python3 train_downstream.py --binary -c ELKI2name --backbone vampnet-best --csv_path data/annotations/allcodas_with_noise.csv  --seed 1

python3 train_downstream.py --binary -c ELKI2name --backbone vampnet-best --csv_path data/annotations/allcodas_with_noise.csv  --seed 2


# Rhythm Type Classification (6 min for each vampnet experiment)

python3 train_downstream.py -c ELKI2name --top_classes 5 --backbone aves --random_init

python3 train_downstream.py -c ELKI2name --top_classes 5 --backbone aves --seed 0

python3 train_downstream.py -c ELKI2name --top_classes 5 --backbone aves --seed 1

python3 train_downstream.py -c ELKI2name --top_classes 5 --backbone aves --seed 2

python3 train_downstream.py -c ELKI2name --top_classes 5 --backbone vampnet-best --grad-accum-steps 2 --seed 0

python3 train_downstream.py -c ELKI2name --top_classes 5 --backbone vampnet-best --grad-accum-steps 2 --seed 1

python3 train_downstream.py -c ELKI2name --top_classes 5 --backbone vampnet-best --grad-accum-steps 2 --seed 2

# Social Unit Classification (18 min for each vampnet experiment)

python3 train_downstream.py -c Unit --backbone aves --random_init

python3 train_downstream.py -c Unit --backbone aves --seed 0

python3 train_downstream.py -c Unit --backbone aves --seed 1

python3 train_downstream.py -c Unit --backbone aves --seed 2

python3 train_downstream.py -c Unit --backbone vampnet-best --seed 0

python3 train_downstream.py -c Unit --backbone vampnet-best --seed 1

python3 train_downstream.py -c Unit --backbone vampnet-best --seed 2

# Vowel Type Classification (3 min for each vampnet)

python3 train_downstream.py -c handv --backbone aves --random_init

python3 train_downstream.py -c handv --backbone aves --seed 0

python3 train_downstream.py -c handv --backbone aves --seed 1

python3 train_downstream.py -c handv --backbone aves --seed 2

python3 train_downstream.py -c handv --backbone vampnet-best --seed 0

python3 train_downstream.py -c handv --backbone vampnet-best --seed 1

python3 train_downstream.py -c handv --backbone vampnet-best --seed 2
