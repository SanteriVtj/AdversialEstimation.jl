function NN_prediction(
    X, Y, Xtest, Ytest;
    chain=SimpleChain(
        static(1),
        TurboDense(tanh, 64),
        TurboDense(tanh, 64),
        TurboDense(tanh, 32),
        TurboDense(identity, 1)
      ),
      epoch=5, iters=10_000
)
    mlpd = chain
  
    p = SimpleChains.init_params(mlpd);
    G = SimpleChains.alloc_threaded_grad(mlpd);
    
    mlpdloss = SimpleChains.add_loss(mlpd, SquaredLoss(Y));
    mlpdtest = SimpleChains.add_loss(mlpd, SquaredLoss(Ytest));
    
    report = let mtrain = mlpdloss, X=X, Xtest=Xtest, mtest = mlpdtest
      p -> begin
        let train = mlpdloss(X, p), test = mlpdtest(Xtest, p)
          @info "Loss:" train test
        end
      end
    end
    
    report(p)
    for _ in 1:epoch
      SimpleChains.train_unbatched!(
        G, p, mlpdloss, X, SimpleChains.ADAM(), iters
      );
      report(p)
    end
    return mlpd(Xtest,p)'
  end