from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
app = Flask(__name__, template_folder = "templates")

from Application import routes