SNR=30
EPOCHS=10
VARIANT='ConvNet'
RECURRENT='NoRec'
#SEQUENCES=false
CV_FILTERS=60
FC_NEURONS=2048

for nb_cv in 1 2 3; do
    for nb_fc in 1 2 3; do
        python3 bench_train.py -bench=1 \
#        python3 bench_plot.py -bench=1 \
#        python3 bench_comp.py -bench=1 \
            -var=$VARIANT -rec=$RECURRENT -snr=$SNR -eps=$EPOCHS --no-seq \
            -cv_layers=$nb_cv -fc_layers=$nb_fc \
            --conv_filters=$CV_FILTERS --linear_neurons=$FC_NEURONS;
    done;
done;
