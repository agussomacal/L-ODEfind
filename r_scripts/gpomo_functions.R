
library("GPoM")


# ============================================ #
#             Auxiliary functions
# ============================================ #
get_dynamic_eq <- function(){
  data("allMod_nVar3_dMax2")
  return(allMod_nVar3_dMax2)
}

get_dynamic_ts <- function(){
  data("TSallMod_nVar3_dMax2")
  
  series <- list()
  for(model_name in names(TSallMod_nVar3_dMax2)){
    series[[model_name]] <- TSallMod_nVar3_dMax2[[model_name]]$reconstr
  }
  
  return(series)
}


get_statistics <- function(series){append(c(mean(series), sd(series), mean(diff(series)), sd(diff(series))), series[1:100])}


fit_gpomo <- function(eq, ts, poly_degree, base_filename, nPmin=NULL, nPmax=NULL, full_information=FALSE){
    if(full_information){
        nS = c(1,1,1)
        nVar = sum(nS)
        where <- seq(2, 4)
    }else{
        nS <- 3
        nVar <- 3
        where <- 2
    }
  
  out <- gPoMo(data =ts[1:1000,where],
               tin =ts[1:1000,1],
               dMax =poly_degree, 
               nS=nS,
               show =1,
               IstepMin =10, 
               IstepMax =15000,
               nPmin =nPmin, 
               nPmax =nPmax,
               method ='rk4')
  
  
  print('best model')
  loss <- list()
  # compare the statistics to the true series to know which model to choose.
  true_statistics <- get_statistics( ts[,where])
  for(model_name in names(out$stockoutreg)){
    loss[[model_name]] <- sum((get_statistics(out$stockoutreg[[model_name]]) - true_statistics)^2)
  }
  
  res_rows <- poLabs(nVar, dMax = poly_degree)
  if((length(out$stockoutreg) == 0) || (all(unlist(lapply(loss, is.na))))){
      print('No model found')
      res <- matrix(0, ncol = 3, nrow = length(res_rows))
      return(res)
  }else{
      print('portrait')
      png(file=paste0(base_filename, "phase_portrait.png"), width = 500, height = 500)
      plot(out$stockoutreg[[which.min(loss)]][,1], out$stockoutreg[[which.min(loss)]][,2],
           type='l',col ='red',xlab ='y(t)',ylab ='dy(t)/dt',main ='Phase portrait')
      lines(out$filtdata[,1], out$filtdata[,2])
      dev.off()
      
      res<- out$models[[which.min(loss)]]
      rownames(res) <- res_rows
      # colnames(res) <- c('dX1/dt', 'dX2/dt', 'dX3/dt')    
  }
  
  return(res)
}


predict <- function(true_eq, fit_eq, ts, poly_degree, Isteps){
  nS = c(1,1,1)
  nVar = sum(nS)
  where <- seq(2, nVar+1)
  
  dt <- ts[2,1]-ts[1,1]
  outNumi <-numicano(nVar=nVar, 
                     dMax=poly_degree, 
                     Istep = Isteps, 
                     onestep = dt, 
                     KL = as.matrix(fit_eq), 
                     v0 = ts[1,where])
  
  outNumi_true <-numicano(nVar=3, 
                          dMax=poly_degree, 
                          Istep = Isteps, 
                          onestep = dt, 
                          KL = as.matrix(true_eq), 
                          v0 = ts[1,where])
  
  return(list(true_model=outNumi_true$reconstr, fitted_model=outNumi$reconstr))
}

plot_predictions <- function(dynamic_system_name, outNumi_true, outNumi){
  par(mfrow=c(1,3))
  for(i in seq(3)){
    plot(outNumi[,1], outNumi[,i+1],type ='l', xlab ='t', ylab ='',main =dynamic_system_name,col ='red')
    lines(outNumi_true[,1], outNumi_true[,i+1], type ='l', xlab ='t', ylab ='', main =dynamic_system_name,col ='blue')
    legend('topright', c("fitted model","true model"),col=c('red','blue'),lty=1,cex =0.8)
  }
  # for(i in seq(3)){
  #   plot(outNumi[,i+1], outNumi_true[,i+1],type ='p', pch=19, cex=0.1, xlab ='true model', ylab ='fitted model',main =dynamic_system_name,col ='black')
  # }
}



