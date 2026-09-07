The data used to create figure 1 and for the magenta colored points in figure 2
can be found in the .vtk files. An NR in the filename indicates, that calculation was performed without
regularization. The files with Fux in the title contain the diffusion coefficient and
the files with reg_potential in the title contain the therivative of the effective potential
with the chemichical potential contribution being subtracted (u-4Δµ²). The masses were calculated
with a forward derivative. 
Each of the .vtk fiels contains a header of 5 lines, where the RG time can be found in line two. The three
columns afterwards are the grid value (Δ), the respective quantity stored in that file, and the last column 
contains only zeros for plotting purposes. At the end of each .vtk file are two additional lines due to the format.
For Figure 3 (a), there are Figure_3_a_1.txt and Figure_3_a_2.txt. The figure was created by merging both of them.