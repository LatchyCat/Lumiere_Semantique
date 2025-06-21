# In ~/lumiere_semantique/backend/lumiere_core/urls.py
# In lumiere_core/urls.py

from django.contrib import admin
from django.urls import path, include # <-- Make sure 'include' is imported

urlpatterns = [
    path('admin/', admin.site.urls),

    # This line tells Django that any URL starting with 'api/v1/'
    # should be handled by the URL patterns defined in our 'api.urls' file.
    path('api/v1/', include('api.urls')),
]
