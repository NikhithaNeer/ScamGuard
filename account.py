from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from functools import wraps
import auth as auth_module
import hashlib

account = Blueprint("account", __name__, template_folder="templates")

# ------------------ 装饰器 ------------------
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in first!", "error")
            return redirect(url_for("auth.login"))
        return f(*args, **kwargs)
    return wrapper

# ------------------ 简单哈希 ------------------
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# ------------------ 编辑账户 ------------------
@account.route("", methods=["GET", "POST"])
@login_required
def edit_account():
    user_id = session.get("user_id")
    supabase = auth_module.supabase
    if not supabase:
        flash("Supabase not initialized!", "error")
        return redirect(url_for("auth.login"))

    try:
        # 获取用户信息
        user_resp = supabase.table("login").select("Name, Email").eq("id", user_id).execute()
        if not user_resp.data:
            flash("User not found!", "error")
            return redirect(url_for("check_job"))

        user = user_resp.data[0]

        if request.method == "POST":
            new_name = request.form.get("username", "").strip()
            new_email = request.form.get("email", "").strip()
            new_password = request.form.get("password", "").strip()
            confirm_password = request.form.get("confirm_password", "").strip()

            updates = {}

            if new_name and new_name != user.get("Name"):
                updates["Name"] = new_name
                session["username"] = new_name  # 更新 session
            if new_email and new_email != user.get("Email"):
                updates["Email"] = new_email
            if new_password:
                if new_password != confirm_password:
                    flash("Passwords do not match!", "error")
                    return render_template("edit_account.html", user=user)
                updates["Password"] = hash_password(new_password)

            if updates:
                supabase.table("login").update(updates).eq("id", user_id).execute()
                flash("Account updated successfully!", "success")
            else:
                flash("No changes detected.", "info")

            # 刷新用户信息
            user_resp = supabase.table("login").select("Name, Email").eq("id", user_id).execute()
            user = user_resp.data[0]

        return render_template("edit_account.html", user=user)

    except Exception as e:
        flash(f"Error: {e}", "error")
        return redirect(url_for("check_job"))
    
# ------------------ 管理员编辑任意用户 ------------------
@account.route("/edit/<int:user_id>", methods=["GET", "POST"])
@login_required
def edit_user(user_id):
    # 确认是管理员
    if int(session.get("permission_level", 0)) < 1:
        flash("Access denied: admin privileges required.", "error")
        return redirect(url_for("dashboard"))

    supabase = auth_module.supabase
    if not supabase:
        flash("Supabase not initialized!", "error")
        return redirect(url_for("auth.login"))

    try:
        # 获取被编辑的用户信息
        user_resp = supabase.table("login").select("id, Name, Email, Permission_level").eq("id", user_id).execute()
        if not user_resp.data:
            flash("User not found!", "error")
            return redirect(url_for("admin_dashboard"))

        user = user_resp.data[0]

        # 管理员不能编辑其他管理员
        if user["Permission_level"] >= 1:
            flash("Admin accounts cannot be edited.", "info")
            return redirect(url_for("admin_dashboard"))

        if request.method == "POST":
            new_name = request.form.get("username", "").strip()
            new_email = request.form.get("email", "").strip()
            new_permission = int(request.form.get("permission_level", "0"))
            new_password = request.form.get("password", "").strip()

            updates = {}
            if new_name and new_name != user.get("Name"):
                updates["Name"] = new_name
            if new_email and new_email != user.get("Email"):
                updates["Email"] = new_email
            if new_permission != user.get("Permission_level"):
                updates["Permission_level"] = new_permission
            if new_password:
                updates["Password"] = hash_password(new_password)

            if updates:
                supabase.table("login").update(updates).eq("id", user_id).execute()
                flash("User updated successfully!", "success")
            else:
                flash("No changes detected.", "info")

            return redirect(url_for("admin_dashboard"))

        return render_template("edit_user.html", user=user)

    except Exception as e:
        flash(f"Error: {e}", "error")
        return redirect(url_for("admin_dashboard"))