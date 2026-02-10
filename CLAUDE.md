Programming Guidelines
0. **Required:** Before developing any code, write a plan in a Markdown file, and place all documentation in the `doc` directory;
1. **Required:** Before developing any specific function, evaluate the input and output, expected behavior, and develop test cases; place all test cases in the `test` directory;
2. **Required:** After developing each function, you must perform testing. If the test fails, analyze the cause of the failure. If the existing information is insufficient to analyze the cause, you can search online for more information. Improve the code based on the analysis until the test passes;
3. **Required:** After completing each small, incremental task, you must use `git commit` to submit the relevant files;
4. **Recommended:** Break down large tasks into smaller tasks, complete the smaller tasks one by one, and check the completeness of the larger task;
5. **Forbidden:** Using simple, non-equivalent tasks to replace complex tasks;
6. **Recommended:** After completing a task, review whether there is a better and simpler solution, and use simple, equivalent tasks to replace complex tasks;
7. **Required**: Use conda environment former3d in /home/cwh/miniconda3/envs/former3d to execute all python related job. 
8. **Forbidden:** Creation of any simplified versions.
9. **Forbidden:** Duplicate code.
10. **Required:** After completing each task, follow this workflow:
    - First, commit the current work progress with `git commit`
    - Then, clean up all newly created intermediate/debug files that are not needed
    - Keep only the final effective new files
    - Finally, commit the clean state with `git commit` again

## Memory Management and Training Configuration

11. **Memory Analysis Required**: Before training with new configurations, always run memory profiling with `--enable-memory-profile` flag. Analyze results using `analyze_memory.py` to identify bottlenecks.

12. **StreamFusion Optimization**: Use concat + MLP fusion (`StreamConcatFusion`) instead of attention mechanisms to reduce memory usage by 99%. This is critical for batch_size=2 training.

13. **Batch Size Guidelines**:
    - **Batch Size 2 (Multi-GPU)**: Recommended when sufficient GPU memory is available (>8GB per GPU). Requires larger crop_size and proper voxel_size settings.
    - **Batch Size 1 (Single-GPU)**: Use when GPU memory is limited (<8GB). Requires modifying 3D network BatchNorm → InstanceNorm to support batch_size=1.

14. **Multi-GPU Training**: Use `--multi-gpu` flag to enable DataParallel on multiple GPUs. Note:
    - Model parameters are duplicated on each GPU
    - Batch is split across GPUs for forward pass
    - GPU 0 handles aggregation and gradient synchronization
    - Memory overhead: ~50-100 MB per GPU

15. **Parameter Tuning for Batch Size 2**:
    - **Conservative Configuration**: `--batch-size 2 --crop-size "6,6,4" --voxel-size 0.30 --accumulation-steps 4`
      Expected memory: ~4-5 GB
    - **Balanced Configuration**: `--batch-size 2 --crop-size "8,8,6" --voxel-size 0.25 --accumulation-steps 2`
      Expected memory: ~6-7 GB
    - **Aggressive Configuration**: `--batch-size 2 --crop-size "10,10,8" --voxel-size 0.20 --accumulation-steps 1`
      Expected memory: ~8-9 GB (may OOM)

16. **Memory Bottleneck Analysis**: When OOM occurs, identify which component causes the issue:
    - **StreamFusion**: Attention matrix calculation → Use concat fusion
    - **3D Sparse Convolution**: Deep layer operations → Reduce crop_size or increase voxel_size
    - **History Buffer**: Accumulating features → Limit cached frames
    - **Build Map Table**: Hash table construction → Reduce voxel count

17. **Optimization Priorities**:
    - **Short-term**: Adjust crop_size, voxel_size, accumulation_steps
    - **Medium-term**: Implement mixed precision training (AMP), gradient checkpointing
    - **Long-term**: Use DistributedDataParallel (DDP), optimize 3D network architecture

18. **Common Errors and Solutions**:
    - **BatchNorm Error (batch_size=1)**: Modify 3D network to use InstanceNorm instead of BatchNorm
    - **CUDA OOM in StreamFusion**: Use `StreamConcatFusion` instead of `StreamCrossAttention`
    - **CUDA OOM in 3D Convolution**: Reduce crop_size or increase voxel_size
    - **SPConv N=0 Error**: Increase crop_size or decrease voxel_size to ensure sufficient voxels

19. **Memory Monitoring During Training**:
    - Enable real-time memory profiling: `--enable-memory-profile --memory-profile-output "memory_analysis"`
    - Monitor GPU memory usage with `nvidia-smi -l 1`
    - Check memory profiler JSON outputs for detailed layer-by-layer analysis
    - Use `analyze_memory.py` to generate human-readable reports

20. **Training Principles Summary**:
    - **Start Conservative**: Begin with smaller configurations that are guaranteed to run
    - **Gradually Scale**: Increase crop_size or reduce voxel_size incrementally
    - **Profile Early**: Run memory profiling early in development to catch issues
    - **Document Results**: Record successful and failed configurations in doc/MEMORY_*.md
    - **Iterate Quickly**: Use gradient accumulation to simulate larger batch sizes without OOM