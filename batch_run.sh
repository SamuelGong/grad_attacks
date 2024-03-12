#!/bin/bash
#for i in {1..10}
#do
#   python main.py 1_gpt2_gen/wikitext_2_tag_s42_it1000_lr05.yml
#done

#python main.py 1_gpt2_gen/wikitext_2_tag_s42_it1000_lr05.yml
#python main.py 1_gpt2_gen/wikitext_2_tag_s42_it1000_lr01.yml
#python main.py 1_gpt2_gen/wikitext_2_tag_s42_it1000_lr005.yml
#python main.py 1_gpt2_gen/wikitext_2_tag_s42_it1000_lr001.yml
#python main.py 1_gpt2_gen/wikitext_2_tag_s42_it1000_lr0005.yml
#python main.py 1_gpt2_gen/wikitext_2_tag_s42_it1000_lr0001.yml
#python main.py 1_gpt2_gen/wikitext_2_tag_s42_it1000_lr00005.yml
#python main.py 1_gpt2_gen/wikitext_2_tag_s42_it1000_lr0001.yml

myArray=(
"2_gpt2_cls/glue_cola_tag_s42_it20000_lr005_s04.yml"
"2_gpt2_cls/glue_cola_tag_s42_it20000_lr005_s02.yml"
"2_gpt2_cls/glue_cola_tag_s42_it20000_lr005_s005.yml"
"2_gpt2_cls/glue_cola_tag_s42_it20000_lr005_s0025.yml"
)

# Iterate over the array
for index in "${!myArray[@]}"; do
  item="${myArray[index]}"
  index=$((index + 1))
  echo "Starting the $index task $item"
  python main.py $item
  # Add your desired operations for each item in the array
done