SNR=30
EPOCHS=10
VARIANT='ConvNet'
RECURRENT='NoRec'
#SEQUENCES=false
NB_CV=2
NB_FC=2

for cv_filters in 20 40 60; do
    for fc_neurons in 512 1024 2048; do
        python3 bench_train.py -bench=2 \
#        python3 bench_plot.py -bench=2 \
#        python3 bench_comp.py -bench=2 \
            -var=$VARIANT -rec=$RECURRENT -snr=$SNR -eps=$EPOCHS --no-seq \
            -cv_layers=$NB_CV -fc_layers=$NB_FC \
            --conv_filters=$cv_filters --linear_neurons=$fc_neurons;
    done;
done;
