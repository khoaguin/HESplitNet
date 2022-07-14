from pkg_resources import declare_namespace
import torch
import tenseal as ts
import sys

BATCH_SIZE = 4

he_context = {
    "P": 8192,  # polynomial_modulus_degree
    "C": [40, 21, 21, 21, 40],  # coeff_modulo_bit_sizes
    "Delta": pow(2, 21)  # the global scaling factor
}
context = ts.context(
    ts.SCHEME_TYPE.CKKS, 
    poly_modulus_degree=he_context["P"], 
    coeff_mod_bit_sizes=he_context["C"]
)
context.global_scale = he_context["Delta"]

W = torch.randn([5,256])
Wt = W.T
b = torch.randn([5])
print(f'Wt: {type(Wt)}, {Wt.shape}')
print(f'b: {type(b)}, {b.shape}')
print()

enc_Wt = ts.CKKSTensor(context=context, tensor=Wt, batch=True)
enc_Wt = enc_Wt.reshape([1, enc_Wt.shape[0]])
print(f'enc_Wt (with batching): {type(enc_Wt)}, {enc_Wt.shape}')
print(f'size of serialized enc_Wt using batching: {sys.getsizeof(enc_Wt.serialize()) / 10**6} Mb')

enc_Wt_nobatch = ts.CKKSTensor(context, Wt)
print(f'enc_Wt (without batching): {type(enc_Wt_nobatch)}, {enc_Wt_nobatch.shape}')
print(f'size of serialized enc_Wt without batching: {sys.getsizeof(enc_Wt_nobatch.serialize()) / 10**6} Mb')
print()

a_t = torch.randn([BATCH_SIZE, 256]).T  # 4 is the batch size
print(f'a_t: {type(a_t)}, {a_t.shape}')
enc_a_t = ts.CKKSTensor(context=context, tensor=a_t, batch=True)
enc_a_t = enc_a_t.reshape([1, enc_a_t.shape[0]])
print(f'enc_a_t: {type(enc_a_t)}, {enc_a_t.shape}')
print(f'size of serialized enc_a_t: {sys.getsizeof(enc_a_t.serialize()) / 10**6} Mb')
print()

temp =  enc_a_t.transpose()
enc_a2 = temp.mm(enc_Wt) + b
print(f'enc_a2: {type(enc_a2)}, {enc_a2.shape}')
print(f'size of serialized enc_a2: {sys.getsizeof(enc_a2.serialize()) / 10**6} Mb')
print()

dJda2 = torch.randn([BATCH_SIZE, 5])
print(f'dJda2 shape: {dJda2.shape}')
enc_dJdWt = enc_a_t.mm(dJda2)
print(f'enc_dJdWt: {enc_dJdWt.shape}')
enc_updated_Wt = enc_Wt - enc_dJdWt
print(f'size of serialized updated & encrypted Wt: {sys.getsizeof(enc_updated_Wt.serialize()) / 10**6} Mb')
decrypted_updated_Wt = torch.tensor(enc_updated_Wt.decrypt().tolist()).squeeze(dim=1)
print(f'decrypted_updated_Wt: {decrypted_updated_Wt.shape}')

