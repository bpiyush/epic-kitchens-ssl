# Runs ablation on number of training samples

model="R2PLUS1D"
for num in {1000,2000,4000,8000,16000,32000}
do
    echo "-------------------------- Running $model with $num samples --------------------"
    cfg="configs/EPIC-KITCHENS/$model/n_samples_"$num"_32x112x112_R18_K400_LR0.0025.yaml"

    echo "Config: "
    ls $cfg
    echo ""

    # bash das5_run_train.sh -c $cfg
    # bash das5_val.sh -c $cfg

    echo "-------------------------- xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx --------------------"
    echo ""
done
