from django.shortcuts import render, redirect
from django.utils import timezone
from .models import Post, FirstVisit, CurrentViewCheck

from django.shortcuts import render, get_object_or_404

from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login, logout, authenticate
from django.contrib import messages

from django.http import HttpResponse
# from django.core import serializers

# Create your views here.

def getcount1(request, pk):
    if request.user.is_authenticated:
        if request.method == "GET":
            post = get_object_or_404(Post, pk=pk)
            return HttpResponse(post.current_view)
        else:
            pass
    else:
        return redirect('/')

def getcount2(request, pk):
    if request.user.is_authenticated:
        if request.method == "GET":
            post = get_object_or_404(Post, pk=pk)
            return HttpResponse(post.blog_views)
        else:
            pass
    else:
        return redirect('/')

def viewpost(request, pk):
    if request.user.is_authenticated: 
        if request.method == "GET":
            if not CurrentViewCheck.objects.filter(user=request.user.username, storyno=request.path.split('/')[-2]).exists(): 
                CurrentViewCheck(user=request.user.username, storyno=request.path.split('/')[-2]).save()
                post = get_object_or_404(Post, pk=pk)
                post.current_view = post.current_view + 1
                post.save()
                # json_serializer = serializers.get_serializer("json")()
                # display_value = json_serializer.serialize(get_object_or_404(Post, pk=pk), ensure_ascii=False)
                # display_value = json_serializer.serialize(post, ensure_ascii=False)
                # display_value = json_serializer.serialize(Post.objects.all(), ensure_ascii = False)
                # display_value = serializers.serialize('json', [post,])
                return HttpResponse(post.current_view)
            else:
                # print(type((request.path.split('/')[-2])))
                print("Alreaady Viewing The Story In Another Tab")
                post = get_object_or_404(Post, pk=pk)
                return HttpResponse(post.current_view)
        else:
            return HttpResponse("Not GET Request")
    else:
        return redirect('/')

def viewpostdecrement(request, pk):
    if request.user.is_authenticated:
        if request.method == "GET":
            if CurrentViewCheck.objects.filter(user=request.user.username, storyno=request.path.split('/')[-2]).exists(): 
                CurrentViewCheck.objects.filter(user=request.user.username, storyno=request.path.split('/')[-2]).delete()
                post = get_object_or_404(Post, pk=pk)
                post.current_view = post.current_view - 1
                post.save()
                return HttpResponse(post.current_view)
            else:
                print(request.path.split('/')[-2])
                post = get_object_or_404(Post, pk=pk)
                return HttpResponse(post.current_view)
        else:
            return HttpResponse("Not GET Request")
    else:
        return redirect('/')

def main(request):
    if not request.user.is_authenticated:
        return render(request, 'mainpage.html', {})
    else:
        return redirect('/home')

def post_list(request):
    if request.user.is_authenticated:
        posts = Post.objects.filter(published_date__lte=timezone.now()).order_by('-published_date')
        return render(request, 'blog/post_list.html', {'posts':posts, 'user':request.user.username})
    else:
        return redirect('/')

# def story1(request):
#     return render(request, 'blog/stories/story1.html', {})

# def story2(request):
#     return render(request, 'blog/stories/story2.html', {})

# def story3(request):
#     return render(request, 'blog/stories/story3.html', {})

def post_detail(request, pk):
    if request.user.is_authenticated:
        if not FirstVisit.objects.filter(user=request.user.id, url=request.path).exists():
            #user visits for the first time, update page count and show
            FirstVisit(user=request.user, url=request.path).save()
            post = get_object_or_404(Post, pk=pk)
            post.blog_views = post.blog_views + 1
            post.save()
            return render(request, 'blog/post_detail.html', {'post': post, 'user':request.user.username})
        else:
            post = get_object_or_404(Post, pk=pk)
            return render(request, 'blog/post_detail.html', {'post': post, 'user':request.user.username})
    else:
        return redirect('/')

def delete_post(request, pk):
    if request.user.is_authenticated:
        Post.objects.filter(id=pk).delete()
        return redirect('/home/')
    else:
        return redirect('/')

def about(request):
    if request.user.is_authenticated:
        return render(request, 'blog/about.html', {'user':request.user.username})
    else:
        return redirect('/aboutme')

def aboutme(request):
    if not request.user.is_authenticated:
        return render(request, 'aboutme.html', {})
    else:
        return redirect('/about')

def register(request):
    if not request.user.is_authenticated:
        if request.method == "POST":
            form = UserCreationForm(request.POST)
            if form.is_valid():
                user = form.save()
                username = form.cleaned_data.get('username')
                messages.success(request, f"New Account Created: {username}")
                login(request, user)
                return redirect("/home")
            else:
                for msg in form.error_messages:
                    messages.error(request, form.error_messages[msg])

        form = UserCreationForm
        return render(request, 'register.html', {'form':form})
    else:
        return redirect('/home')

def logout_request(request):
    logout(request)
    messages.info(request, "Logged Out Successfully")
    return redirect('/')

def login_request(request):
    if not request.user.is_authenticated:
        if request.method == "POST":
            form = AuthenticationForm(request, data=request.POST)
            if form.is_valid():
                username = form.cleaned_data.get('username')
                password = form.cleaned_data.get('password')
                user = authenticate(username=username, password=password)
                if user is not None:
                    login(request, user)
                    messages.success(request, f"Successfully Logged In As {username}")
                    return redirect('/home')
                else:
                    messages.error(request, "Invalid Username or Password")
            else:
                    messages.error(request, "Invalid Username or Password")
    else:
        return redirect('/home')

    form = AuthenticationForm()
    return render(request, "login.html", {"form": form})
