
gp_reg_64_to_32 = {"%rax": "%eax",
                   "%rcx": "%ecx",
                   "%rdx": "%edx",
                   "%rbx": "%ebx",
                   "%rsp": "%esp",
                   "%rbp": "%ebp",
                   "%rsi": "%esi",
                   "%rdi": "%edi",
                   "%r8": "%r8d",
                   "%r9": "%r9d",
                   "%r10": "%r10d",
                   "%r11": "%r11d",
                   "%r12": "%r12d",
                   "%r13": "%r13d",
                   "%r14": "%r14d",
                   "%r15": "%r15d",
                   }

gp_reg_64_to_16 = {'%rax': '%ax',
                     '%rcx': '%cx',
                     '%rdx': '%dx',
                     '%rbx': '%bx',
                     '%rsp': '%sp',
                     '%rbp': '%bp',
                     '%rsi': '%si',
                     '%rdi': '%di',
                     '%r8': '%r8w',
                     '%r9': '%r9w',
                     '%r10': '%r10w',
                     '%r11': '%r11w',
                     '%r12': '%r12w',
                     '%r13': '%r13w',
                     '%r14': '%r14w',
                     '%r15': '%r15w'}

gp_reg_64_to_8 = {'%rax': '%al',
                     '%rcx': '%cl',
                     '%rdx': '%dl',
                     '%rbx': '%bl',
                     '%rsp': '%sil',
                     '%rbp': '%bpl',
                     '%rsi': '%dil',
                     '%rdi': '%r8b',
                     '%r8': '%r9b',
                     '%r9': '%r10b',
                     '%r10': '%r11b',
                     '%r11': '%r12b',
                     '%r12': '%r13b',
                     '%r13': '%r14b',
                     '%r14': '%r15b'}

gp_reg_32_to_16 = {"%eax": "%ax",
                   "%ecx": "%cx",
                   "%edx": "%dx",
                   "%ebx": "%bx",
                   "%esp": "%sp",
                   "%ebp": "%bp",
                   "%esi": "%si",
                   "%edi": "%di",
                   "%r8d": "%r8w",
                   "%r9d": "%r9w",
                   "%r10d": "%r10w",
                   "%r11d": "%r11w",
                   "%r12d": "%r12w",
                   "%r13d": "%r13w",
                   "%r14d": "%r14w",
                   "%r15d": "%r15w",
                   }

gp_reg_16_to_8 = {"%ax": "%al",
                   "%cx": "%cl",
                   "%dx": "%dl",
                   "%bx": "%bl",
                   "%sp": "%sil",
                   "%bp": "%bpl",
                   "%si": "%sil",
                   "%di": "%dil",
                   "%r8w": "%r8b",
                   "%r9w": "%r9b",
                   "%r10w": "%r10b",
                   "%r11w": "%r11b",
                   "%r12w": "%r12b",
                   "%r13w": "%r13b",
                   "%r14w": "%r14b",
                   "%r15w": "%r15b",
                   }

gp_reg_8_to_None = {"%al": None,
                   "%cl": None,
                   "%dl": None,
                   "%bl": None,
                   "%sil": None,
                   "%bpl": None,
                   "%sil": None,
                   "%dil": None,
                   "%r8b": None,
                   "%r9b": None,
                   "%r10b": None,
                   "%r11b": None,
                   "%r12b": None,
                   "%r13b": None,
                   "%r14b": None,
                   "%r15b": None,
                   }

simd_reg_256_to_128 = {"%ymm0": "%xmm0",
                       "%ymm1": "%xmm1",
                       "%ymm2": "%xmm2",
                       "%ymm3": "%xmm3",
                       "%ymm4": "%xmm4",
                       "%ymm5": "%xmm5",
                       "%ymm6": "%xmm6",
                       "%ymm7": "%xmm7",
                       "%ymm8": "%xmm8",
                       "%ymm9": "%xmm9",
                       "%ymm10": "%xmm10",
                       "%ymm11": "%xmm11",
                       "%ymm12": "%xmm12",
                       "%ymm13": "%xmm13",
                       "%ymm14": "%xmm14",
                       "%ymm15": "%xmm15"}

simd_reg_128_to_None = {"%xmm0": None,
                        "%xmm1": None,
                        "%xmm2": None,
                        "%xmm3": None,
                        "%xmm4": None,
                        "%xmm5": None,
                        "%xmm6": None,
                        "%xmm7": None,
                        "%xmm8": None,
                        "%xmm9": None,
                        "%xmm10": None,
                        "%xmm11": None,
                        "%xmm12": None,
                        "%xmm13": None,
                        "%xmm14": None,
                        "%xmm15": None,
}

flags_to_none = {"%cf": None,
                 "%pf": None,
                 "%af": None,
                 "%zf": None,
                 "%sf": None,
                 "%tf": None,
                 "%if": None,
                 "%df": None,
                 "%of": None,
                 "%nt": None,
                 "%rf": None,
                 "%vm": None,
                 "%ac": None,
                 "%vif": None,
                 "%vip": None,
                 "%id": None}

REGISTER_TO_STDOUT_REGISTER = {
                   "%rax": "%rax",
                   "%rcx": "%rcx",
                   "%rdx": "%rdx",
                   "%rbx": "rbx",
                   "%rsp": "%rsp",
                   "%rbp": "%rbp",
                   "%rsi": "%rsi",
                   "%rdi": "%rdi",
                   "%r8": "%r8",
                   "%r9": "%r9",
                   "%r10": "%r10",
                   "%r11": "%r11",
                   "%r12": "%r12",
                   "%r13": "%r13",
                   "%r14": "%r14",
                   "%r15": "%r15",

                   "%eax": "%rax",
                   "%ecx": "%rcx",
                   "%edx": "%rdx",
                   "%ebx": "%rbx",
                   "%esp": "%rsp",
                   "%ebp": "%rbp",
                   "%esi": "%rsi",
                   "%edi": "%rdi",
                   "%r8d": "%r8",
                   "%r9d": "%r9",
                   "%r10d": "%r10",
                   "%r11d": "%r11",
                   "%r12d": "%r12",
                   "%r13d": "%r13",
                   "%r14d": "%r14",
                   "%r15d": "%r15",

                   "%ax": "%rax",
                   "%cx": "%rcx",
                   "%dx": "%rdx",
                   "%bx": "%rbx",
                   "%sp": "%rsp",
                   "%bp": "%rbp",
                   "%si": "%rsi",
                   "%di": "%rdi",
                   "%r8w": "%r8",
                   "%r9w": "%r9",
                   "%r10w": "%r10",
                   "%r11w": "%r11",
                   "%r12w": "%r12",
                   "%r13w": "%r13",
                   "%r14w": "%r14",
                   "%r15w": "%r15",

                   "%al": "%rax",
                   "%cl": "%rcx",
                   "%dl": "%rdx",
                   "%bl": "%rbx",
                   "%sil": "%rsp",
                   "%bpl": "%rbp",
                   "%sil": "%rsi",
                   "%dil": "%rdi",
                   "%r8b": "%r8",
                   "%r9b": "%r9",
                   "%r10b": "%r10",
                   "%r11b": "%r11",
                   "%r12b": "%r12",
                   "%r13b": "%r13",
                   "%r14b": "%r14",
                   "%r15b": "%r15",
                    "%ymm0": "%ymm0",
                    "%ymm1": "%ymm1",
                    "%ymm2": "%ymm2",
                    "%ymm3": "%ymm3",
                    "%ymm4": "%ymm4",
                    "%ymm5": "%ymm5",
                    "%ymm6": "%ymm6",
                    "%ymm7": "%ymm7",
                    "%ymm8": "%ymm8",
                    "%ymm9": "%ymm9",
                    "%ymm10": "%ymm10",
                    "%ymm11": "%ymm11",
                    "%ymm12": "%ymm12",
                    "%ymm13": "%ymm13",
                    "%ymm14": "%ymm14",
                    "%ymm15": "%ymm15",
                    "%xmm0": "%ymm0",
                    "%xmm1": "%ymm1",
                    "%xmm2": "%ymm2",
                    "%xmm3": "%ymm3",
                    "%xmm4": "%ymm4",
                    "%xmm5": "%ymm5",
                    "%xmm6": "%ymm6",
                    "%xmm7": "%ymm7",
                    "%xmm8": "%ymm8",
                    "%xmm9": "%ymm9",
                    "%xmm10": "%ymm10",
                    "%xmm11": "%ymm11",
                    "%xmm12": "%ymm12",
                    "%xmm13": "%ymm13",
                    "%xmm14": "%ymm14",
                    "%xmm15": "%ymm15",

                 "%cf": "%cf",
                 "%pf": "%pf",
                 "%af": "%af",
                 "%zf": "%zf",
                 "%sf": "%sf",
                 "%tf": "%tf",
                 "%if": "%if",
                 "%df": "%df",
                 "%of": "%of",
                 "%nt": "%nt",
                 "%rf": "%rf",
                 "%vm": "%vm",
                 "%ac": "%ac",
                 "%vif": "%vif",
                 "%vip": "%vip",
                 "%id": "%id"

                   }


SIMD_REGISTERS = list(simd_reg_128_to_None.keys())
GP_REGISTERS = list(gp_reg_64_to_32.keys())
LIVE_OUT_FLAGS = ["%zf", "%cf"]
DEF_IN_FLAGS = ["%zf", "%cf", "%mxcsr::rc[0]"]

SIMD_REGISTERS_SET = set(SIMD_REGISTERS)
GP_REGISTERS_SET = set(GP_REGISTERS)
DEF_IN_FLAGS_SET = set(DEF_IN_FLAGS)

LIVE_OUT_REGISTER_LIST = GP_REGISTERS + SIMD_REGISTERS + LIVE_OUT_FLAGS
DEF_IN_REGISTER_LIST = GP_REGISTERS + SIMD_REGISTERS + DEF_IN_FLAGS


NEXT_REGISTER_TESTING_DICT = {**gp_reg_64_to_32, **gp_reg_32_to_16, **gp_reg_16_to_8, **gp_reg_8_to_None,
                              **simd_reg_256_to_128, **simd_reg_128_to_None, **flags_to_none}
