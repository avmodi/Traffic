class ModelRunner(object):
    
  
     def run(self, model_class, trials=3, log=False, read_file=None, limit=1):
        
        if log:
            old_stdout = sys.stdout
            log_name = self.save_file.replace('results', 'logs')[:-4]   
            log_file = open(log_name + time.strftime("%x").replace('/', '.') + '.txt', 'w', buffering=1)
            sys.stdout = log_file
        self.cresults = self._read_results()
        unsuccessful_settings = []
        for params in self.param_list:
            for data in self.data_list:
                if limit < np.inf:
                    already_computed = self.lookup_setting(read_file=read_file,
                                                           params=params, data=data,
                                                           irrelevant=irrelevant)
                    if already_computed >= limit:
                        print('Found %d (>= limit = %d) computed results for the setting:' % (already_computed, limit))
                        for k, v in params.items():
                            print(str(k).rjust(15) + ': ' +  str(v))
                        continue
                    else:
                        required_success = limit - already_computed
                        print('Found %d (< limit = %d) computed results for the setting.' % (already_computed, limit))
                else:
                    required_success = 1
                success, errors = 0, 0
                setting_time = time.time()
                while (errors < trials) and (success < required_success):
#                    try:
                    print('As yet, for this configuration: success: %d, errors: %d' % (success, errors))
                    for k, v in params.items():
                        print(str(k).rjust(15) + ': ' +  str(v))
                    self.cdata = data
                    self.cp = params
                    print("using " + repr(model_class) + " to build the model")
                    model = model_class(data, params, os.path.join(WDIR, 'tensorboard'))
#                    history, nn, reducer = model.run()
                    model_results, nn = model.run()
                    self.nn = nn
#                    self.reducer = reducer
                    model_results.update(params)
                    hdf5_name = self._get_hdf5_name()
                    print('setting time %.2f' % (time.time() - setting_time))
                    nn.save(hdf5_name)
                    model_results.update(
                        {'training_time': time.time() - setting_time,
                         'datetime': datetime.datetime.now().isoformat(),
                         'dt': datetime.datetime.now(),
                         'date': datetime.date.today().isoformat(),
                         'data': data,
                         'hdf5': hdf5_name,
                         'total_params': np.sum([np.sum([np.prod(K.eval(w).shape) for w in l.trainable_weights]) for l in nn.layers])
#                             'json': nn.to_json(),
#                             'model_params': reducer.saved_layers
                         }
                    )
                    self.cresults.append(model_results)
                    pd.DataFrame(self.cresults).to_pickle(self.save_file)
                    success += 1
#                    except Exception as e:
#                        errors += 1
#                        print(e)
                if success < required_success:
                    unsuccessful_settings.append([data, params])
        #    with open(save_file, 'wb') as f:
        #        pickle.dump(results, f)
        with open(self.save_file[:-4] + 'failed.pikle', 'wb') as f:
            pickle.dump(unsuccessful_settings, f)
        if log:
            sys.stdout = old_stdout
            log_file.close()
        return self.cresults
        
