Readme file with instructions

conda env create -f environment.yaml python=3.6.7

----------------------------------------------------------------------------------------
To train models:

cd transformers/examples
========================================================================================
To train : title to storyline


python run_language_modeling.py --output_dir=./ROC-STORYLINE/ --model_type=gpt2 --model_name_or_path=gpt2 --do_train --train_data_file=../storyline-data/train.txt --do_eval --eval_data_file=../storyline-data/test.txt --eval_all_checkpoints --num_train_epochs=20 --line_by_line --per_gpu_train_batch_size=3 --per_gpu_eval_batch_size=3 --evaluate_during_training

=========================================================================================

To train : storyline to story


python run_language_modeling.py --output_dir=./ROC-STORY/ --model_type=gpt2 --model_name_or_path=gpt2 --do_train --train_data_file=../story-data/train.txt --do_eval --eval_data_file=../story-data/test.txt --eval_all_checkpoints --num_train_epochs=20 --line_by_line --per_gpu_train_batch_size=3 --per_gpu_eval_batch_size=3 --evaluate_during_training

--------------------------------------------------------------------------------------------
Please download zip files storyline-cp.zip and story-cp.zip from the drive links and place under transformers folder. These are pre-trained models necessary to run in interactive mode

To run in interactive mode 
cd transformers/
============================================================================================

To generate storyline from title


If you want to just generate one instance
python generate-title_to_storyline.py False "big break ====== "

If you want to just generate for a file
python generate-title_to_storyline.py True "storyline-data/test.txt"

==========================================================================================

To generate story from title+storyline


If you want to just generate one instance
python generate-title+storyline_to_story.py False "big break ====== dreams singer # practiced # greet # people # amazement wanted %%%% "

If you want to just generate for a file
python generate-title+storyline_to_story.py True "story-data/test.txt"

==========================================================================================
To generate story from title

If you want to just generate one instance
python generate-title_to_story.py False "big break ====== "

If you want to just generate for a file
python generate-title_to_story.py True "storyline-data/test.txt"
