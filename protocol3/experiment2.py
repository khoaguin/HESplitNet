import torch
import tenseal as ts
import sys

BATCH_SIZE = 64

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
Wt = W.T  # [256,5]
b = torch.randn([5])
print(f'Wt: {type(Wt)}, {Wt.shape}')
print(f'b: {type(b)}, {b.shape}')
print()

enc_Wt = ts.CKKSTensor(context=context, tensor=Wt, batch=True)
enc_enc_Wt = ts.CKKSVector(context=context, vector=enc_Wt)
# enc_Wt = enc_Wt.reshape([1, enc_Wt.shape[0]])
# print(f'enc_Wt (with batching): {type(enc_Wt)}, {enc_Wt.shape}')
# print(f'size of serialized enc_Wt using batching: {sys.getsizeof(enc_Wt.serialize()) / 10**6} Mb')

# enc_Wt_nobatch = ts.CKKSTensor(context, Wt)
# print(f'enc_Wt (without batching): {type(enc_Wt_nobatch)}, {enc_Wt_nobatch.shape}')
# print(f'size of serialized enc_Wt without batching: {sys.getsizeof(enc_Wt_nobatch.serialize()) / 10**6} Mb')
# print()

a = torch.randn([BATCH_SIZE, 256])
print(f'a: {type(a)}, {a.shape}')
enc_a = ts.CKKSTensor(context=context, tensor=a, batch=True)
enc_a = enc_a.reshape([1, enc_a.shape[0]])
print(f'enc_a: {type(enc_a)}, {enc_a.shape}')
print(f'size of serialized enc_a: {sys.getsizeof(enc_a.serialize()) / 10**6} Mb')
print()

a_flat = a.flatten()
enc_a_flat = ts.CKKSTensor(context=context, tensor=a_flat, batch=True)
print(f'enc_a_flat: {type(enc_a_flat)}, {enc_a_flat.shape}')
print(f'size of serialized enc_a_flat: {sys.getsizeof(enc_a_flat.serialize()) / 10**6} Mb')

enc_a2 = enc_a.mm(Wt) + b
print(f'enc_a2: {type(enc_a2)}, {enc_a2.shape}')
print(f'size of serialized enc_a2: {sys.getsizeof(enc_a2.serialize()) / 10**6} Mb')
print()

dec_a2 = enc_a2.decrypt()
print(f'dec_a2: {type(dec_a2)}, {dec_a2.shape}')
a2 = torch.tensor(dec_a2.tolist())
a2 = a2.squeeze()
print(a2.shape)

dJda2 = torch.randn([BATCH_SIZE, 5])
print(f'dJda2 shape: {dJda2.shape}')
# enc_dJdWt = enc_a.mm(dJda2)
# print(f'enc_dJdWt: {enc_dJdWt.shape}')
# enc_updated_Wt = enc_Wt - enc_dJdWt
# print(f'size of serialized updated & encrypted Wt: {sys.getsizeof(enc_updated_Wt.serialize()) / 10**6} Mb')
# decrypted_updated_Wt = torch.tensor(enc_updated_Wt.decrypt().tolist()).squeeze(dim=1)
# print(f'decrypted_updated_Wt: {decrypted_updated_Wt.shape}')

