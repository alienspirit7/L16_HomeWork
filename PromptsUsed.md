# Session Prompts & Outcomes

## User Prompts (verbatim excerpts)
1. “Create a PRD for the script that does the following. It takes .csv file as an input. The file has a two colomns table: title string, group string. The script takes the titles, converts them into embeddings, then normalise the vectors. It should keep the information of their original groups. Another script takes all the title all together (their normalized vectors) and runs K-Means (K=3) on them and split them into 3 groups. It should then show original titles (their text form) with their original groups and new, K-Mean defined, groups. Another script Asks user to input the title, run it through K-Means based data (vectors and their K-Mean based groups) and define with KNN (K=3) which group the new title belongs the best.”  
2. “embedding model  - all-MiniLM-L6-v2. The file was added to the L16_HomeWork folder. Create PRD in md format”  
3. “Use the PDR to actually create all the scripts.”  
4. “Add requirements.txt file and ReadMe.md file with clear explanations on how run the program (add script that will utilize/orchestrate all of them if needed). Add file structure to the readMe.md.”  
5. “add gitignore file to exclude sensitive files and folders”  
6. “add myenv folder”  
7. “Do you see the issues I had when running in Terminal?”  
8. “What does that mean: /Users/alienspirit/Documents/25D/L16_HomeWork/myenv/lib/python3.9/site-packages/sklearn/utils/extmath.py:203: RuntimeWarning: divide by zero encountered in matmul…”  
9. “Add visuals to Kmeans results on how this looks like visually. Make the titles related visuals of different colors as per their originals groups”  
10. “Add additional option on how titles converted to embeddings, using gemini API.”  
11. “Add to the batch_predictions file, what were the nearest K-Means based groups (0,1,2) and distanced to them (similar to what you did with original groups).”  
12. “How should I store my gemini key?”  
13. “Where in env folder do I store the key and how do I use it>”  
14. “yes, lets do that config file please”  
15. “It looks it still runs embedding through non gemini model”  
16. “Check the error again, its something else, not internet”  
17. “Add pip install google-generativeai to requirements file”  
18. “Check all the images in the Images folder again and review Readme and incorporate all the images there”  
19. “Based on the work done, update PRD, add tasks.json file, create Prompts used file with prompts from this session and summary of what was done”  
20. “Add the last prompt to the PromtesUsed file”  
21. “Update PromptsUsed file again with all the actual prompts from this ssession (quote them)”  
22. “Add .md file with results analysis that will go through the example we did and summarise how K-Means made mistakes and allocated some title to a wrong groups (especially when regular model was used for embeddings). discuss why such mistakes could happen. Will compare the 2 embeddings models and underline good and bad sides of each model usage. Define how K-Means dependent on the model chosen for embeddings. Discuss why even though the gemini model was much more accurate, why the result of the tested new title allocation to cluster wasn't good.”  
23. “Check all the images in the Images folder again and review Readme and incorporate all the images there” *(repeat instruction for completeness)*.  
24. “Gemini didn't identify messi title correctly. It was defined as Women Soccer instead. Please correct”  
25. “Add to lessons and recommendations that having more titles in training set can help with future titles classifications”  
26. “Update prompts file with recent prompts added”  

## Work Summary
- Authored a comprehensive PRD covering SentenceTransformer and Gemini embedding paths, reporting, classification, visualization, and orchestration requirements.  
- Implemented and iteratively refined the CLI scripts (`prepare_embeddings.py`, `run_kmeans.py`, `classify_title.py`, `visualize_clusters.py`, `run_pipeline.py`) with robust parsing, manifest metadata, config support, and enriched outputs.  
- Produced documentation: README with dual-provider instructions, workflow diagram, embedded process screenshots, and dependency list (including `google-generativeai`).  
- Added developer tooling (`.vscode/tasks.json`) for common tasks and ensured Git hygiene (`.gitignore`, config handling).  
- Investigated runtime issues (CSV casing, silhouette warnings, vector serialization, Gemini setup) and implemented fixes or guidance.  
- Delivered this prompts log summarizing user interactions and completed deliverables.
