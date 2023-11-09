#!/bin/bash

#!/bin/bash

number=$1
md=$2
dm=$3

if [ "$md" -eq 1 ]; then
    Method="--Method TMTGP"
elif [ "$md" -eq 2 ]; then
    Method="--Method WS"
elif [ "$md" -eq 3 ]; then
    Method="--Method BO"
elif [ "$md" -eq 4 ]; then
    Method="--Method INC"
elif [ "$md" -eq 5 ]; then
    Method="--Method Meta"
elif [ "$md" -eq 6 ]; then
    Method="--Method TPE"
elif [ "$md" -eq 7 ]; then
    Method="--Method HEBO"
elif [ "$md" -eq 8 ]; then
    Method="--Method RF"
elif [ "$md" -eq 9 ]; then
    Method="--Method MT"
elif [ "$md" -eq 10 ]; then
    Method="--Method abl1"
elif [ "$md" -eq 11 ]; then
    Method="--Method abl2"
elif [ "$md" -eq 12 ]; then
    Method="--Method abl3"
elif [ "$md" -eq 13 ]; then
    Method="--Method abl4"
else
    echo "无效的参数"
    exit 1
fi

if [ "$dm" -eq 1 ]; then
    dim="--dim 2"
elif [ "$dm" -eq 2 ]; then
    dim="--dim 5"
elif [ "$dm" -eq 3 ]; then
    dim="--dim 8"
elif [ "$dm" -eq 4 ]; then
    dim="--dim 10"
else
    echo "无效的参数"
    exit 1
fi

# 输出数值范围
for ((i=start; i<=end; i++))
do

	seed="--Seed $number"
	python3 run.py $seed $method $dim &
	# 将Python程序的PID保存到数组pids中
	pids[$i]=$!
	echo "第 $i 次Python程序已启动，PID为: ${pids[$i]}"
done

for ((i=start; i<=end; i++))
do
	wait ${pids[$i]}
	echo "第 $i 次Python程序已结束"
done

echo "所有Python程序已结束"

