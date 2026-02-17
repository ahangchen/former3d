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
11. **Required:** Place all test results, test logs in the `test_results` directory;
12. **Required:** Place all test reports in the `test_reports` directory;
13. **Forbidden**: Use batch size 1 for any training, validation or test;
14. **Required**: Place all auxiliary tool code (such as training monitoring, resource monitoring) and scripts in the utils directory.
