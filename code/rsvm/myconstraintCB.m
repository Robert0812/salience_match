function yhat = myconstraintCB(param, model, x, y)
% slack resaling: argmax_y delta(yi, y) (1 + <psi(x,y), w> - <psi(x,yi), w>)
% margin rescaling: argmax_y delta(yi, y) + <psi(x,y), w>
%   if dot(y*x, model.w) > 1, yhat = y ; else yhat = - y ; end
%   if param.verbose
%     fprintf('yhat = violslack([%8.3f,%8.3f], [%8.3f,%8.3f], %3d) = %3d\n', ...
%             model.w, x, y, yhat) ;
%   end
    
    % y     = x
    % x     = current sample
    % model = current model
    % param = context parameter
    % yhat  = most violated constraint
    
    if all(model.w == 0)
        model.w = ones(length(model.w), 1);
        yhat = circshift(y, [1, -1]);
        
    else
        pos             = param.pos{x};
        neg             = param.neg{x};
        ScorePos        = cellfun(@(z) dot(model.w, z(:)), ...
            param.phi(x, pos));
        [Vpos, Ipos]    = sort(full(ScorePos'), 'descend');
        Ipos            = pos(Ipos);
        
        ScoreNeg        = cellfun(@(z) dot(model.w, z(:)), ...
            param.phi(x, neg));
        [Vneg, Ineg]    = sort(full(ScoreNeg'), 'descend');
        Ineg            = neg(Ineg);
        
        numPos          = length(pos);
        numNeg          = length(neg);
        n               = numPos + numNeg;
        
        NegsBefore = sum(bsxfun(@lt, Vpos, Vneg), 1);
%         NegsBefore = max(0, NegsBefore-10); 
%         if NegsBefore < n
%             in = randperm(min(n - NegsBefore, 10));
%             NegsBefore = NegsBefore + in(1);
%         end
        
        yhat        = nan * ones(n, 1);
        yhat((1:numPos) + NegsBefore) = Ipos;
        yhat(isnan(yhat)) = Ineg;
        
%         yhat = y;
%         
%         beta = 0.1;
%         neg_beta_floor = floor(numNeg*beta);
%         fplus = ScorePos;
%         fminus = ScoreNeg;
%         fminus(Ineg(1:neg_beta_floor)) = fminus(Ineg(1:neg_beta_floor))+1;
%         [~, fsort_indices] = sort([fminus, fplus], 'descend');
%         labels = [-ones(numNeg, 1); ones(numPos, 1)]; %param.label{x};
%         slabels = labels(fsort_indices);
%         yhat = convert2output(slabels, Ipos, Ineg, numPos, numNeg);
    end
    
end

function a = convert2output(labels, pos_ind, neg_ind, numPos, numNeg)    
    % CONVERT GIVEN OUTPUT TO A REPRESENTATION OF SIZE mxn
    
      % Global variables visible to other functions
      m = numPos;
      n = numNeg;

      % Maintain number of positive and negative examples seen so far
      i = 0; % Number of positive and negative examples seen so far
      j = 0; % Number of positive and negative examples seen so far

      % a = [aplus; aminus]
      aplus = zeros(m, 1);
      aminus = zeros(n, 1);

      pos = 0; % No. of positives encountered
      neg = 0; % No. of negatives encountered

      for l = labels'
          if(l == 1) %Positive label
              i = i + 1;
              pos = pos + 1;
              aplus(pos_ind(i)) = n - neg;
          else %Negative label
              j = j + 1;
              neg = neg + 1;
              aminus(neg_ind(j)) = pos;
          end
      end

      a = [aplus; aminus];
end
