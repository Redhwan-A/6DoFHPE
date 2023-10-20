# HPE
Frm-Hpe: Full-Range Markerless Head Pose Estimation

# Citing

If you find our work useful, please cite the paper:



\begin{table}[t]
\centering
\caption{Mean average errors of the predicted Euler angles in the CMU benchmark datasets.}
\label{tabe1}
\begin{tabular}{lccccc}
\hline
\multicolumn{1}{l|}{\textbf{Method}} & \multicolumn{1}{c|}{\textbf{Retrain?}} & \textbf{Yaw}  & \textbf{Pitch} & \textbf{Roll} & \textbf{MAE}  \\ \hline
\multicolumn{6}{c}{Narrow-range: -90◦ \textless yaw \textless 90◦}                                                                             \\ \hline
\multicolumn{1}{l|}{WHENet}          & \multicolumn{1}{c|}{No}                & 37.96         & 22.7           & 16.54         & 25.73         \\
\multicolumn{1}{l|}{HopeNet}         & \multicolumn{1}{c|}{No}                & 20.40         & 17.47          & 13.40         & 17.09         \\
\multicolumn{1}{l|}{FSA-Net}         & \multicolumn{1}{c|}{No}                & 17.52         & 16.31          & 13.02         & 15.62         \\
\multicolumn{1}{l|}{DirectMHP}       & \multicolumn{1}{c|}{Yes}               & 5.86          & 8.25           & 7.25          & 7.12          \\
\multicolumn{1}{l|}{DirectMHP}       & \multicolumn{1}{c|}{No}                & 5.75          & 8.01           & 6.96          & 6.91          \\
\multicolumn{1}{l|}{6DRepNet}        & \multicolumn{1}{c|}{Yes}               & 5.20          & 7.22           & 6.00          & 6.14          \\
\multicolumn{1}{l|}{Frm-Hpe (ours)}   & \multicolumn{1}{c|}{Yes}               & \textbf{5.13} & \textbf{6.99}  & \textbf{5.77} & \textbf{5.96} \\ \hline
\multicolumn{6}{c}{Full-range: -180◦ \textless yaw \textless 180◦}                                                                             \\ \hline
\multicolumn{1}{l|}{WHENet}          & \multicolumn{1}{c|}{No}                & 29.87         & 19.88          & 14.66         & 21.47         \\
\multicolumn{1}{l|}{DirectMHP}       & \multicolumn{1}{c|}{Yes}               & 7.38          & 8.56           & 7.47          & 7.80          \\
\multicolumn{1}{l|}{DirectMHP}       & \multicolumn{1}{c|}{No}                & 7.32          & 8.54           & 7.35          & 7.74          \\
\multicolumn{1}{l|}{6DRepNet}        & \multicolumn{1}{c|}{Yes}               & 5.89          & 7.76           & 6.39          & 6.68          \\
\multicolumn{1}{l|}{Frm-Hpe (ours)}   & \multicolumn{1}{c|}{Yes}               & \textbf{5.83} & \textbf{7.63}  & \textbf{6.35} & \textbf{6.60} \\ \hline
\end{tabular}
\end{table}


