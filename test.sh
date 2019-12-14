python3 train.py datasets

python3 train2.py train configs/magic-point_coco_train.yaml magic-point_coco

python train2.py train configs/magic-point_shapes.yaml magic-point_synth

primitive = "magic-point_synth"
tar_path = "datasets"
dataset.dump_primitive_data(self, primitive, tar_path, config)

python experiment.py train configs/magic-point_shapes.yaml magic-point_synth-test
