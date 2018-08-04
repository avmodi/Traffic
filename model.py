
class model():
	
	def build(self):
		"""
		Function returns a keras model object
		"""  
		self.input_length=[10]
		self.output_length=[1]

		# network structure definition
		inp = Input(shape=(self.input_length, self.idim), dtype='float32', name='inp')
		value_input = Input(shape=(self.input_length, self.odim), dtype='float32', name='value_input')
		
		offsets, sigs, loop_layers = [inp], [inp], {}
		
		for j in range(self.layers_no['sigs']):
			# significance sub-network
			name = 'significance' + str(j+1)
			ks = self.kernelsize[j % len(self.kernelsize)] if (type(self.kernelsize) == list) else self.kernelsize
			# Convolution layer
			loop_layers[name] = Conv1D(
				self.filters if (j < self.layers_no['sigs'] - 1) else self.odim, 
				kernel_size=ks, padding='same', 
				activation='linear', name=name,
				kernel_constraint=maxnorm(self.norm)
			)
			sigs.append(loop_layers[name](sigs[-1]))
			
			loop_layers[name + 'BN'] = BatchNormalization(name=name + 'BN')
			sigs.append(loop_layers[name + 'BN'](sigs[-1]))
			
			# residual connections for ResNet
			if self.resnet and (self.connection_freq > 0) and (j > 0) and ((j+1) % self.connection_freq == 0) and (j < self.layers_no['sigs'] - 1):
				sigs.append(keras.layers.add([sigs[-1], sigs[-3 * self.connection_freq + (j==1)]], 
													name='significance_residual' + str(j+1)))
						   
			loop_layers[name + 'act'] = LeakyReLU(alpha=.1, name=name + 'act') if (self.act == 'leakyrelu') else Activation(self.act, name=name + 'act')
			sigs.append(loop_layers[name + 'act'](sigs[-1]))
		
		for j in range(self.layers_no['offs']):
			# offset sub-network
			name = 'offset' + str(j+1)
			loop_layers[name] = Conv1D(
				self.filters if (j < self.layers_no['offs'] - 1) else self.odim,
				kernel_size=1, padding='same', 
				activation='linear', name=name,
				kernel_constraint=maxnorm(self.norm)
			)
			offsets.append(loop_layers[name](offsets[-1]))
			
			loop_layers[name + 'BN'] = BatchNormalization(name=name + 'BN')
			offsets.append(loop_layers[name + 'BN'](offsets[-1]))
			
			
			if self.resnet and (self.connection_freq > 0) and (j > 0) and ((j+1) % self.connection_freq == 0) and (j < self.layers_no['offs'] - 1):
				offsets.append(keras.layers.add([offsets[-1], offsets[-3 * self.connection_freq + (j==1)]], 
													   name='offset_residual' + str(j+1)))
							
			loop_layers[name + 'act'] = LeakyReLU(alpha=.1, name=name + 'act') if (self.act == 'leakyrelu') else Activation(self.act, name=name + 'act')
			offsets.append(loop_layers[name + 'act'](offsets[-1]))
			
		value_output = keras.layers.add([offsets[-1], value_input], name='value_output')
	
		value = Permute((2,1))(value_output)
	
		
		sig = Permute((2,1))(sigs[-1])
		if self.architecture['softmax']:    
			sig = TimeDistributed(Activation('softmax'), name='softmax')(sig)
		elif self.architecture['lambda']:    
			sig = TimeDistributed(Activation('softplus'), name='relulambda')(sig)
			sig = TimeDistributed(Lambda(lambda x: x/K.sum(x, axis=-1, keepdims=True)), name='lambda')(sig)
			
		main = keras.layers.multiply(inputs=[sig, value], name='significancemerge')
		if self.shared_final_weights:
			out = TimeDistributed(Dense(self.output_length, activation='linear', use_bias=False,
										kernel_constraint=nonneg() if self.nonnegative else None),
								  name= 'out')(main)
		else: 
			outL = LocallyConnected1D(filters=1, kernel_size=1,   									  padding='valid')
			out = outL(main)
			
		main_output = Permute((2,1), name='main_output')(out)
		
		nn = keras.models.Model(inputs=[inp, value_input], outputs=[main_output, value_output])


		nn.compile(optimizer=keras.optimizers.Adam(lr=self.lr, clipnorm=self.clipnorm),
				   loss={'main_output': 'mse', 'value_output' : 'mse'},
				   loss_weights={'main_output': 1., 'value_output': self.aux_weight}) 
	
		return nn
