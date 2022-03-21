# Runs ablation on number of training samples

model="VIDEOMOCO"
base_cfg_name="32x112x112_VMOCO_R18_K400_LR0.0025.yaml"
for num in {1000,2000,4000,8000,16000,32000}
do
    echo "-------------------------- Running $model with $num samples --------------------"
    cfg="configs/EPIC-KITCHENS/$model/diva_n_samples_"$num"_"$base_cfg_name

    echo "Config: "
    ls $cfg
    echo ""

    bash diva_train.sh -c $cfg
    bash diva_val.sh -c $cfg

    echo "-------------------------- xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx --------------------"
    echo ""
done
