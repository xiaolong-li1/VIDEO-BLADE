cd /workspace/Vbench_EVA/cogvideo_batch_sampler/simple
#1:激活环境，需要修改
source /workspace/Vbench_EVA/vbench_env/bin/activate
#2:采样过程 要新建一个配置文件，参考8.29_cogvideo4steps_2.json，修改内部的path，

# naming_prompt_file"/workspace/Vbench_EVA/VBench/prompts/all_dimension.txt"
# sampling_prompt_file"/workspace/Vbench_EVA/VBench/prompts/augmented_prompts/gpt_enhanced_prompts/all_dimension_longer.txt"
# 这俩只用把VBench的位置改了后边半截不用动

python simple_multiprocess_sampler.py --config configs/8.29_cogvideo4steps_2.json
#3:评估脚本
#记得激活对应的环境 修改脚本内部的output_base_dir，他会在这个路径下新建一个文件夹存放评估结果
bash /workspace/Vbench_EVA/cogvideo_batch_sampler/simple/VBench/vbench2_beta_long/evaluate_long.sh “videopath” 

#4 计算得分 例子 利用本路径下的calc_finnal_score计算 --result_dir为上一步存放评估结果的路径
python /workspace/Vbench_EVA/cogvideo_batch_sampler/simple/calc_finnal_score.py --result_dir /workspace/Vbench_EVA/cogvid
eo_batch_sampler/simple/Cogvideo5b_4steps_0.84/evaluate_result/checkpoint_120_fake_without_sparsity
#传入上一步我们的采样视频文件夹路径 也记得修改这个脚本里的output_base_dir，评测结果会在那里存放
# python simple_multiprocess_sampler.py --config configs/8.29_cogvideo4steps_1.json