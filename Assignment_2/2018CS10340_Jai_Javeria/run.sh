#    ./run.sh 1 <path_of_train_data> <path_of_test_data>  <part_num>
if [ $1 -eq 1 ]
then
	echo "q1 called"
	if [ $4 = "a" ]
	then
		echo "part a called"
		python ngrams.py $2 $3 1 0
	fi
	if [ $4 = "b" ]
	then
		echo "part b called"
		python q1b.py $2 $3
	fi
	if [ $4 = "c" ]
	then
		echo "part c called"
		python printConfusionMat.py $2 $3
	fi

fi
