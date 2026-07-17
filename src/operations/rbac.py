from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RBACError(PermissionError):
    """Raised when a role tries to perform an unauthorized operation."""


class Role(str, Enum):
    OPERATOR = "operator"
    MAINTENANCE = "maintenance"
    QA = "QA"
    MANAGER = "manager"


class Permission(str, Enum):
    VIEW_RCA = "view_rca"
    CLOSE_ACTIONS = "close_actions"
    APPROVE_ACTIONS = "approve_actions"
    VIEW_REPORTS = "view_reports"


ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.OPERATOR: frozenset(
        {
            Permission.VIEW_RCA,
            Permission.VIEW_REPORTS,
        }
    ),
    Role.MAINTENANCE: frozenset(
        {
            Permission.VIEW_RCA,
            Permission.CLOSE_ACTIONS,
            Permission.VIEW_REPORTS,
        }
    ),
    Role.QA: frozenset(
        {
            Permission.VIEW_RCA,
            Permission.CLOSE_ACTIONS,
            Permission.APPROVE_ACTIONS,
            Permission.VIEW_REPORTS,
        }
    ),
    Role.MANAGER: frozenset(
        {
            Permission.VIEW_RCA,
            Permission.CLOSE_ACTIONS,
            Permission.APPROVE_ACTIONS,
            Permission.VIEW_REPORTS,
        }
    ),
}


@dataclass(frozen=True)
class AccessDecision:
    allowed: bool
    role: Role
    permission: Permission
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "role": self.role.value,
            "permission": self.permission.value,
            "reason": self.reason,
        }


def parse_role(role: str | Role) -> Role:
    if isinstance(role, Role):
        return role

    value = str(role).strip()
    if value.lower() == "qa":
        return Role.QA

    for r in Role:
        if r.value.lower() == value.lower():
            return r
    valid = ", ".join([r.value for r in Role])
    raise ValueError(f"Unknown role '{role}'. Expected one of: {valid}")


def parse_permission(permission: str | Permission) -> Permission:
    if isinstance(permission, Permission):
        return permission

    value = str(permission).strip().lower()
    for p in Permission:
        if p.value == value:
            return p
    valid = ", ".join([p.value for p in Permission])
    raise ValueError(f"Unknown permission '{permission}'. Expected one of: {valid}")


def allowed_permissions(role: str | Role) -> set[Permission]:
    parsed_role = parse_role(role)
    return set(ROLE_PERMISSIONS.get(parsed_role, frozenset()))


def has_permission(role: str | Role, permission: str | Permission) -> bool:
    parsed_role = parse_role(role)
    parsed_permission = parse_permission(permission)
    return parsed_permission in ROLE_PERMISSIONS.get(parsed_role, frozenset())


def evaluate_access(role: str | Role, permission: str | Permission) -> AccessDecision:
    parsed_role = parse_role(role)
    parsed_permission = parse_permission(permission)

    allowed = has_permission(parsed_role, parsed_permission)
    if allowed:
        reason = f"Role '{parsed_role.value}' is authorized for '{parsed_permission.value}'."
    else:
        reason = f"Role '{parsed_role.value}' is not authorized for '{parsed_permission.value}'."

    return AccessDecision(
        allowed=allowed,
        role=parsed_role,
        permission=parsed_permission,
        reason=reason,
    )


def enforce_permission(role: str | Role, permission: str | Permission, *, context: str | None = None) -> None:
    decision = evaluate_access(role, permission)
    if decision.allowed:
        return

    suffix = f" Context: {context}" if context else ""
    raise RBACError(f"Access denied: {decision.reason}.{suffix}")


# Convenience checks for the 4 required enforcement points
def can_view_rca(role: str | Role) -> bool:
    return has_permission(role, Permission.VIEW_RCA)


def can_close_actions(role: str | Role) -> bool:
    return has_permission(role, Permission.CLOSE_ACTIONS)


def can_approve_actions(role: str | Role) -> bool:
    return has_permission(role, Permission.APPROVE_ACTIONS)


def can_view_reports(role: str | Role) -> bool:
    return has_permission(role, Permission.VIEW_REPORTS)


if __name__ == "__main__":
    # Minimal smoke output for manual runtime validation.
    checks = [
        (Role.OPERATOR, Permission.VIEW_RCA),
        (Role.OPERATOR, Permission.APPROVE_ACTIONS),
        (Role.MAINTENANCE, Permission.CLOSE_ACTIONS),
        (Role.QA, Permission.APPROVE_ACTIONS),
        (Role.MANAGER, Permission.VIEW_REPORTS),
    ]

    for role, perm in checks:
        print(evaluate_access(role, perm).as_dict())