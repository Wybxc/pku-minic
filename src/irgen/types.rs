use std::{fmt::Display, num::NonZeroI32};

use koopa::ir::Type;

/// Value type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VType {
    Int,
    Void,
    Ptr(Box<VType>),
    Array(Box<VType>, NonZeroI32),
}

impl VType {
    pub fn is_int(&self) -> bool {
        matches!(self, VType::Int)
    }

    pub fn is_void(&self) -> bool {
        matches!(self, VType::Void)
    }

    pub fn into_ptr(self) -> Self {
        VType::Ptr(Box::new(self))
    }

    pub fn into_array(self, index: NonZeroI32) -> Self {
        VType::Array(Box::new(self), index)
    }

    pub fn to_indexed(&self) -> IndexCast {
        match self {
            VType::Ptr(ty) => IndexCast::Ptr(*ty.clone()),
            VType::Array(ty, _) => IndexCast::Array(*ty.clone()),
            _ => IndexCast::Fail,
        }
    }

    pub fn cast(&self, other: &Self) -> TypeCast {
        if self == other {
            TypeCast::Noop
        } else {
            match (self, other) {
                (VType::Array(ty1, _), VType::Ptr(ty2)) => {
                    if ty1 == ty2 {
                        TypeCast::ArrayToPtr
                    } else {
                        TypeCast::Fail
                    }
                }
                _ => TypeCast::Fail,
            }
        }
    }

    pub fn to_koopa(&self) -> Type {
        match self {
            VType::Int => Type::get_i32(),
            VType::Void => Type::get_unit(),
            VType::Ptr(ty) => Type::get_pointer(ty.to_koopa()),
            VType::Array(ty, index) => Type::get_array(ty.to_koopa(), index.get() as usize),
        }
    }

    pub fn size(&self) -> Option<usize> {
        match self {
            VType::Int => Some(4),
            VType::Void => Some(0),
            VType::Ptr(_) => Some(4),
            VType::Array(ty, index) => ty
                .size()
                .and_then(|ty_size| ty_size.checked_mul(index.get() as usize)),
        }
    }
}

impl Display for VType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VType::Int => write!(f, "int"),
            VType::Void => write!(f, "void"),
            VType::Ptr(ty) => write!(f, "{}*", ty),
            VType::Array(ty, index) => {
                write!(f, "{}[{}]", ty, index)
            }
        }
    }
}

pub enum TypeCast {
    Noop,
    ArrayToPtr,
    Fail,
}

pub enum IndexCast {
    Ptr(VType),
    Array(VType),
    Fail,
}
