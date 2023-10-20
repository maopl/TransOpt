for ((i=3; i<=7; i++))
do

        seed="--Seed $i"
        python3 run.py $seed 
	# 将Python程序的PID保存到数组pids中
       
        echo "第 $i 次Python程序已启动，PID为: ${pids[$i]}"
done

