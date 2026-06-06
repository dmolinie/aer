SNR=10
EPOCHS=10
#SEQUENCES=false
NB_CV=3
NB_FC=3
CV_FILTERS=60
FC_NEURONS=2048

# Chunks variant
for variant in 'ConvNet' 'SincNet' 'DENet'; do
    for recurrent in 'NoRec' 'LSTM' 'GRU'; do
        python3 bench_train.py -bench=0 \
#        python3 bench_plot.py -bench=0 \
#        python3 bench_comp.py -bench=0 \
            -var=$variant -rec=$recurrent -snr=$SNR -eps=$EPOCHS --no-seq \
            -cv_layers=$NB_CV -fc_layers=$NB_FC \
            --conv_filters=$CV_FILTERS --linear_neurons=$FC_NEURONS;
    done;
done;

## Sequences variant
#for variant in 'ConvNet' 'SincNet' 'DENet'; do
#    for recurrent in 'NoRec' 'LSTM' 'GRU'; do
#        python3 bench_train.py -bench=0 \
##        python3 bench_plot.py -bench=0 \
##        python3 bench_comp.py -bench=0 \
#            -var=$variant -rec=$recurrent -snr=$SNR -eps=$EPOCHS --seq \
#            -cv_layers=$NB_CV -fc_layers=$NB_FC \
#            --conv_filters=$CV_FILTERS --linear_neurons=$FC_NEURONS;
#    done;
#done;

