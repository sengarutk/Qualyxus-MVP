"""Who may do what in the console and on the command line."""
from __future__ import annotations

from enum import Enum


class RBACError(PermissionError):
    """A person tried something their role does not allow."""


class Role(str, Enum):
    OPERATOR = "operator"
    MAINTENANCE = "maintenance"
    QA = "QA"
    MANAGER = "manager"


class Permission(str, Enum):
    VIEW = "view"
    REVIEW_PARTS = "review_parts"        # say whether a sampled part was really good
    CLOSE_ACTIONS = "close_actions"
    APPROVE_ACTIONS = "approve_actions"
    MANAGE_MODELS = "manage_models"      # promote, roll back, approve retraining


_P = Permission
ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.OPERATOR: frozenset({_P.VIEW}),
    Role.MAINTENANCE: frozenset({_P.VIEW, _P.CLOSE_ACTIONS}),
    Role.QA: frozenset({_P.VIEW, _P.REVIEW_PARTS, _P.CLOSE_ACTIONS, _P.APPROVE_ACTIONS, _P.MANAGE_MODELS}),
    Role.MANAGER: frozenset({_P.VIEW, _P.REVIEW_PARTS, _P.CLOSE_ACTIONS, _P.APPROVE_ACTIONS, _P.MANAGE_MODELS}),
}


def parse_role(role: str | Role) -> Role:
    for known in Role:
        if known.value.lower() == str(getattr(role, "value", role)).strip().lower():
            return known
    raise ValueError(f"Unknown role '{role}'. Expected one of: {', '.join(r.value for r in Role)}")


def permissions_of(role: str | Role) -> frozenset[Permission]:
    return ROLE_PERMISSIONS[parse_role(role)]


def enforce(role: str | Role, permission: Permission) -> None:
    if permission not in permissions_of(role):
        raise RBACError(f"Role '{parse_role(role).value}' may not {permission.value.replace('_', ' ')}")
