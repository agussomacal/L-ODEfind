require("GPoM")

mse <- function(x, y){
    means = colMeans(y)
    sds = apply(y,2,sd)
    xnorm = scale(x, center = means, scale = sds)
    ynorm = scale(y, center = means, scale = sds)
    return(mean((xnorm-ynorm)^2))
}
    
            
fit_gpomo <- function(time, ladata, max_time_derivative, poly_degree, steps){
    # output: coefficients of best gpomo model as the minimum mse metric in train prediction series.
    n_vars = dim(ladata)[2]

            out <- gPoMo(data =ladata,
                         tin =time,
                         dMax =poly_degree, 
                         nS=rep(max_time_derivative, n_vars),
                         show =0,
                         IstepMin =10, 
                         IstepMax =steps,
                         nPmin=0, 
                         nPmax=Inf,
                         method ='rk4')

    # calculate mse for all the tested models
    loss <- list()
    for(model_name in names(out$stockoutreg)){
        model_data = out$stockoutreg[[model_name]][,-1]
        data_posta = ladata
        # TODO: filtrar con el tiempo y no quedarse con los priemros como ahora
        if (any(is.na(model_data)) || dim(model_data)[1]<dim(data_posta)[1]){
            loss[[model_name]] <- Inf
        }else{
            colnames(model_data) = colnames(data_posta)
            model_data = model_data[seq(length(time)),]
            loss[[model_name]] <- mse(model_data, data_posta)
        }
    }
    
    # return the coefficients of the model with the minimum value of mse
    # -> matrix
    coefs <- out$models[[names(loss)[which.min(loss)]]]
    colnames(coefs) <- rev(poLabs(nVar=max_time_derivative*n_vars, dMax = 1)[-1])
    rownames(coefs) <- poLabs(nVar=max_time_derivative*n_vars, dMax = poly_degree)
    return(coefs)
        
}



experiment <- function(read_data_folder, write_data_folder, steps_list, max_time_derivative, poly_degree){
    print(read_data_folder)
    dir.create(file.path(write_data_folder), showWarnings = FALSE)
    
    # read data
    fit_time_list <- list()
    for(filename in list.files(path=read_data_folder, full.names=FALSE, recursive=FALSE)){
        if (grepl('solution', filename, fixed = TRUE)){
            print(filename)
            ladata <- read.csv(paste(read_data_folder, filename, sep='/'))
            time <- ladata[,1]
            ladata <- ladata[,-1]
            
            #fit data and time it
            for(steps in steps_list){
                print(paste('Doing steps:', steps))
                t0 <- Sys.time()
                coefs <- fit_gpomo(ladata=ladata,
                                   time=time, 
                                   max_time_derivative=max_time_derivative, 
                                   poly_degree=poly_degree, 
                                   steps=steps)
                initcond = unlist(strsplit(unlist(strsplit(filename, "init_cond_", fixed = TRUE))[2], ".csv", fixed=TRUE))[1]
                params = unlist(strsplit(unlist(strsplit(filename, "params_", fixed = TRUE))[2], "_", fixed=TRUE))[1]
                out_filename <- paste0('solution-gpomo_coefs-vars_x_y_z-dmax_',max_time_derivative,'-poly_',poly_degree,'-params_',params,'-init_cond_',initcond,'-steps_',steps,'.csv')
                
                fit_time_list[[out_filename]] <-  difftime(Sys.time(), t0, units = "secs")
                print(paste('Time:',fit_time_list[[out_filename]]))
                write.csv(fit_time_list, paste(write_data_folder, 'times.csv', sep='/'))

                print('Saving coefs.')
                write.csv(coefs, paste(write_data_folder, out_filename, sep='/'))
            }
        }
    }
}

args <- commandArgs(trailingOnly = TRUE)

print(args)
path_data = '/home/yamila/projects/rte2020/ret-ode/data/'
path_results =  '/home/yamila/projects/rte2020/ret-ode/results2/'
experiment(
    #read_data_folder = paste0(path_data, "LorenzAttractor",args[1],"/"),
    #write_data_folder =paste0(path_results, "LorenzAttractor", args[1],"/"),
    read_data_folder = paste0(path_data, "LorenzAttractor/"),
    write_data_folder =paste0(path_results, "LorenzAttractor_x_y_z/"),
    
    #steps_list=c(40,640,1280,5120,10240),
    steps_list = c(40),
    max_time_derivative=1, 
    poly_degree=2
)





        