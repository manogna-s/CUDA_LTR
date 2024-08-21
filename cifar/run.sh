# python3 main.py --dataset cifar100 --imb_ratio 100 --num_max 500 --epochs 200 --gpu 0 --out ./logs --loss_fn bs --cuda --cutout
python3 main.py --dataset cifar100 --imb_ratio 100 --num_max 500 --epochs 200 --gpu 0 --out ./logs --loss_fn ce --cuda --cutout
python3 main.py --dataset cifar100 --imb_ratio 100 --num_max 500 --epochs 200 --gpu 0 --out ./logs --loss_fn ce_drw --cuda --cutout
python3 main.py --dataset cifar100 --imb_ratio 100 --num_max 500 --epochs 200 --gpu 0 --out ./logs --loss_fn ldam_drw --cuda --cutout
python3 main.py --dataset cifar100 --imb_ratio 100 --num_max 500 --epochs 200 --gpu 0 --out ./logs --loss_fn ride --cuda --cutout
python3 main.py --dataset cifar100 --imb_ratio 100 --num_max 500 --epochs 200 --gpu 0 --out ./logs --loss_fn bcl --cuda --cutout
python3 main.py --dataset cifar100 --imb_ratio 100 --num_max 500 --epochs 200 --gpu 0 --out ./logs --loss_fn ncl --cuda --cutout


