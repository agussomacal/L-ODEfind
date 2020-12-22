require("GPoM")

mse <- function(x, y){
  means = colMeans(y)
  sds = apply(y,2,sd)
  xnorm = scale(x, center = means, scale = sds)
  ynorm = scale(y, center = means, scale = sds)
  return(mean((xnorm-ynorm)^2))
}

smape <- function(x,y){
  return(2/length(x) * sum(abs(x-y)/(abs(x)+abs(y))))
}


fit_gpomo <- function(time, ladata, observed_var, max_time_derivative, poly_degree, steps, plots_filename, verbose = TRUE){
  # output: coefficients of best gpomo model as the minimum smape metric in train prediction series.
    n_vars = length(observed_var)
    
    out <- gPoMo(data = ladata,
                 tin = time,
                 dMax =poly_degree, 
                 nS=rep(max_time_derivative, n_vars),
                 show =0,
                 IstepMin =10, 
                 IstepMax =steps,
                 nPmin=0, 
                 nPmax=Inf,
                 method ='rk4')
    
  if (verbose){
    plots_folder = paste0(plots_filename)
    dir.create(plots_folder, showWarnings = FALSE)
  }
  #visuOutGP(out)
  # calculate mse for all the tested models
  loss <- list()
  for(model_name in names(out$stockoutreg)){
    model_data = out$stockoutreg[[model_name]][,seq(2,n_vars+1,1)] #keep the column corresponding to the observed variables
    data_posta = ladata
    

    # TODO: filtrar con el tiempo y no quedarse con los priemros como ahora
    if (any(is.na(model_data))){
      loss[[model_name]] <- Inf
    }else{
      if  (length(model_data)<length(data_posta)){
        data_posta_loss = data_posta[1:length(model_data)]
        model_data_loss = model_data
      }else{
        model_data_loss = model_data[1:length(time)]
        data_posta_loss = data_posta
      }
      
      #colnames(model_data) = colnames(data_posta)
      loss[[model_name]] <- smape(model_data_loss, data_posta_loss)
    }
    
    if (verbose){
      png(paste0(plots_folder,'/',model_name, '.png'))
      plot(data_posta,col = 'blue', main = paste(model_name,'\n', loss[[model_name]]), type='l')
      lines(model_data, col = 'red')
      dev.off()
    }
  }
  best_model = names(loss)[which.min(loss)]
  #visuEq(out$models[[best_model]], nVar=max_time_derivative, dMax = poly_degree, approx = 2)
  coefs <- out$models[[best_model]]
  colnames(coefs) <- rev(poLabs(nVar=max_time_derivative*n_vars, dMax = 1, Xnote=toupper(observed_var))[-1])
  rownames(coefs) <- poLabs(nVar=max_time_derivative*n_vars, dMax = poly_degree, Xnote=toupper(observed_var))
  return(coefs)
  

}



experiment <- function(read_data_folder, write_data_folder, observed_var, keep2test, steps_list, max_time_derivatives, poly_degrees, verbose){
  print(read_data_folder)
  dir.create(file.path(write_data_folder), showWarnings = TRUE)
  
  # read data
  fit_time_list <- list()
  for(filename in list.files(path=read_data_folder, full.names=FALSE, recursive=FALSE, )){
    if (grepl('solution', filename, fixed = TRUE)){
      print('--------------------------')
      print(filename)
      print('--------------------------')
      ladata_raw <- read.csv(paste(read_data_folder, filename, sep='/'))
      n <- dim(ladata_raw)[1]
      time <- ladata_raw[1:(n-keep2test),1]
      ladata <- ladata_raw[1:(n-keep2test),observed_var]
      
      #fit data and time it
      for(steps in steps_list){
        for (max_time_derivative in max_time_derivatives){
          for (poly_degree in poly_degrees){
            print(paste('Doing steps:', steps, 'max time derivative', max_time_derivative, 'poly degree', poly_degree))
            
  
          # filename 
          out_filename1 <- paste0('gpomo_coefs-vars_', paste(observed_var, collapse='_'),'-dmax_', max_time_derivative, '-poly_', poly_degree,'-')
          data_name = unlist(strsplit(filename, ".", fixed = TRUE))[1]
          init_cond = strsplit(data_name,'init_cond_')[[1]][2]
          params = strsplit(strsplit(data_name,'params_')[[1]][2], '_')[[1]][1]
          out_filename2 <- paste0('params_', params,'-init_cond_', init_cond, '-steps_', steps)
          out_filename <- paste0('solution-', out_filename1, out_filename2, '.csv')
          plots_filename <- paste0(write_data_folder,'plots-',out_filename1, out_filename2)
          
          t0 <- Sys.time()
          
          coeffs <- fit_gpomo(ladata=ladata,
                              observed_var = observed_var,
                              time=time, 
                              max_time_derivative=max_time_derivative, 
                              poly_degree=poly_degree, 
                              steps=steps,
                              plots_filename = plots_filename,
                              verbose = verbose)
          
  
          fit_time_list[[out_filename]] <-  difftime(Sys.time(), t0, units = "secs")
          print(paste('Time:',fit_time_list[[out_filename]]))
          write.csv(fit_time_list, paste(write_data_folder, 'times.csv', sep='/'))
          
          write.csv(coeffs, paste(write_data_folder, out_filename, sep='/'))
          }
        }
      }
    }
  }
}

args <- commandArgs(trailingOnly = TRUE)

read_data_folder = args[1]
write_data_folder = args[2]
observed_var= args[3]
steps = args[4]
print(read_data_folder)
print(write_data_folder)
print(observed_var)
print(steps)

experiment(
    read_data_folder = read_data_folder,
    write_data_folder = write_data_folder,
    observed_var = observed_var,
    keep2test = 200,
    steps_list= steps,
    max_time_derivatives=c(2,3), 
    poly_degrees=c(3),
    verbose = TRUE
)




