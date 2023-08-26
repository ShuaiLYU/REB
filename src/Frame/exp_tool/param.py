"""

20221028


"""
import copy



__all__=["get_common_SGD_optim_param","Param"]
class Param(object):
    def __init__(self, param_name="", **kargs):
        self._param_name = param_name
        self.regist_from_dict(kargs)

    def regist_from_parser(self, parser):
        self.regist_from_dict(parser.__dict__)
        # for key,val in parser.__dict__.items():
        #     self.__setitem__(key, val)

    def regist_from_dict(self, _dict):
        assert isinstance(_dict, dict)
        for key, val in _dict.items():
            self.check_key(key)
            self.__setitem__(key, val)

    def regist_child(self, param_name: str, init_param=None):
        self[param_name] = init_param.clone() if init_param is not None else Param()
        return self[param_name]

    @property
    def name(self):
        name = self._param_name.split(".")[-1]
        return name

    @property
    def param_name(self):
        return self._param_name

    def check_key(self, key):
        assert (key != "param_name")
        assert (key != "name")
        assert (key != "keys")
        assert (key != "vals")
        assert (key != "items")

    def update_name(self, last_name, key):
        self._param_name = last_name + "." + key
        for key, val in self.__dict__.items():
            if isinstance(val, Param):
                val.update_name(self._param_name, key)

    # 功能 A["a"]
    def __setitem__(self, key, value):
        super(Param, self).__setattr__(key, value)
        if isinstance(value, Param):
            value.update_name(self._param_name, key)

    # self.__dict__[key] = value
    def __getitem__(self, attr):
        # print(attr)
        return super(Param, self).__getattribute__(attr)

    def __delitem__(self, key):
        try:
            del self.__dict__[key]
        except KeyError as k:
            return None

    # 功能  A.a
    def __setattr__(self, key, value):
        super(Param, self).__setattr__(key, value)
        if isinstance(value, Param):
            value.update_name(self._param_name, key)

    # self.__dict__[key] = value
    def __getattribute__(self, attr):
        return super(Param, self).__getattribute__(attr)

    # def __getattr__(self, attr):
    # 	"""|
    # 	重载此函数防止属性不存在时__getattribute__报错，而是返回None
    # 	那“_ getattribute_”与“_ getattr_”的最大差异在于：
    # 	1. 无论调用对象的什么属性，包括不存在的属性，都会首先调用“_ getattribute_”方法；
    # 	2. 只有找不到对象的属性时，才会调用“_ getattr_”方法；
    # 	:param attr:
    # 	:return:
    # 	"""
    # 	return super(Param, self).__getattr__(attr)
    # raise Exception("attr:[{}] is not existing".format(attr))
    def __delattr__(self, key):
        try:
            del self.__dict__[key]
        except KeyError as k:
            return None

    # def __str__(self):
    # 	string=""
    # 	for key,val in self.__dict__.items():
    # 		if key is "_name": continue
    # 		if isinstance(val,Param):
    # 			string += self._name + "{}=Param()\n".format(key)
    # 			string +="{}".format(val)
    # 		else:
    # 			string +=self._name+"{}={}\n".format(key,val)
    # 	return string
    def __str__(self):
        string = self._param_name + "=Param()\n"
        for key, val in self.__dict__.items():
            if key == "_param_name": continue
            if isinstance(val, Param):
                string += str(val)
            elif isinstance(val, str):
                string += self._param_name + ".{}='{}'\n".format(key, val)

            else:
                string += self._param_name + ".{}={}\n".format(key, val)
        return string

    def __len__(self):
        return len(self.keys())

    def keys(self):
        keys = [key for key in self.__dict__.keys() if key != "_param_name"]
        return keys

    def values(self):
        return [self[key] for key in self.keys()]

    def items(self):

        return [(key, self[key]) for key in self.keys()]

    def get(self, key, defaut):
        if key in self.keys():
            return self[key]
        else:
            return defaut

    def clone(self):
        return copy.deepcopy(self)

    # 	"""|
    # 	重载此函数防止属性不存在时__getattribute__报错，而是返回None
    # 	那“_ getattribute_”与“_ getattr_”的最大差异在于：
    # 	1. 无论调用对象的什么属性，包括不存在的属性，都会首先调用“_ getattribute_”方法；
    # 	2. 只有找不到对象的属性时，才会调用“_ getattr_”方法；
    # 	:param attr:
    # 	:return:
    # 	"""
    # 	return super(Param, self).__getattr__(attr)
    # raise Exception("attr:[{}] is not existing".format(attr))
    def __delattr__(self, key):
        try:
            del self.__dict__[key]
        except KeyError as k:
            return None

    # def __str__(self):
    # 	string=""
    # 	for key,val in self.__dict__.items():
    # 		if key is "_name": continue
    # 		if isinstance(val,Param):
    # 			string += self._name + "{}=Param()\n".format(key)
    # 			string +="{}".format(val)
    # 		else:
    # 			string +=self._name+"{}={}\n".format(key,val)
    # 	return string
    def __str__(self):
        string = self._param_name + "=Param()\n"
        for key, val in self.__dict__.items():
            if key == "_param_name": continue
            if isinstance(val, Param):
                string += str(val)
            elif isinstance(val, str):
                string += self._param_name + ".{}='{}'\n".format(key, val)

            else:
                string += self._param_name + ".{}={}\n".format(key, val)
        return string

    def __len__(self):
        return len(self.keys())

    def keys(self):
        keys = [key for key in self.__dict__.keys() if key != "_param_name"]
        return keys

    def values(self):
        return [self[key] for key in self.keys()]

    def items(self):
        return [item for item in self if item[0] in self.keys()]

    def get(self, key, defaut):
        if key in self.keys():
            return self[key]
        else:
            return defaut

    def clone(self):
        return copy.deepcopy(self)
    
    

def get_common_SGD_optim_param(optimizer_name):
    assert  optimizer_name in ["SGD","Adam"]
    optim_param = Param(
        optimizer_name=optimizer_name,
        lrReducer_name=None,
        with_earlyStopper=False,
        with_warmUp=False,
    )
    optim_param.SGD = Param(lr=0.1, momentum=0.9, weight_decay=0.001)   ## momentum 0.5->0.1
    optim_param.Adam = Param(lr=0.001, weight_decay=0.001, betas=(0.9, 0.999))
    optim_param.ReduceLROnPlateau = Param(mode="min", verbose=True, patience=3, factor=0.5)
    optim_param.EarlyStopper = Param(thred_loss=0.01, verbose=True, patience=10)
    optim_param.WarmUp = Param(min_lr=1e-6,num_steps=100,method='line',verbose=True)
    return  optim_param
