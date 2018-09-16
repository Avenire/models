[[ $1 > 0 ]] && GPU=$1 || GPU=0
for seed in 1234 123456 12345678 666 87654321 654321 4321
do
./experiment_template.sh $seed new_api_bahdanau_bilstm $GPU
done