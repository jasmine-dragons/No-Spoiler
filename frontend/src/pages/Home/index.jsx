import logo from '../../assets/logo.svg';
import search from '../../assets/search.svg';
import chat from '../../assets/chat-bubble.svg';
import './style.less';
import Comment from '../../components/Comment';
import LargePost from '../../components/LargePost';
import Post from '../../components/Post';
import {useState, useEffect} from 'react';
import { API_URL } from '../../config';

function Home() {
    const [comment, setComment] = useState("");
    const [postList, setPostList] = useState([]);
    const [selectedPost, setSelectedPost] = useState(null);

    const checkSubmission = (e) => {
        if (e.key === 'Enter') {
            fetch(API_URL + '/comment', {
                method: "POST",
                headers: {
                    Accept: "application/json",
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    username: localStorage.getItem("username"),
                    comment: comment
                })
            })
            .then(response => response.json())
            .then(data => console.log(data));
        }
    }

    useEffect(() => {
        fetch(API_URL + '/getposts')
        .then(response => response.json())
        .then(data => {console.log(data); setPostList(data)});
    }, []);

    return (
        <div className="App">
        <header className="App-header">
            <img src={logo} alt="logo"/>
            <p onClick={() => {window.location.href = "/signin"}}>Logout</p>
        </header>
            <div className="App-body">
                <div className="App-left">
                    <div className="search-create-div">
                        <div className="search">
                            <img src={search} alt="search"/>
                            <input type="text" placeholder="Search" />
                        </div>
                        <div className="create" onClick={() => {window.location.href = "/create-post"}}>
                            <img src={chat} alt="chat"/>
                            Create Post
                        </div>
                    </div>
                    {postList.length !== 0 && postList.map((post, index) => {

                        console.log(post);
                        return (
                    <Post
                        username={post.author}
                        title={post.title}
                        onClick={() => {setSelectedPost(post)}}
                        selected={selectedPost && post["_id"]["$oid"] === selectedPost["_id"]["$oid"]}
                    />
                    )})}
                </div>
                {selectedPost && <div className="App-right">
                    <div className="post-comment-div">
                        <LargePost username={selectedPost.author} title={selectedPost.title} body={selectedPost.description} spoiler={selectedPost.isSpoiler}/>
                        <div className="blank"></div>
                        {/* <Comment username="sqirley" comment="that movie was cool"/> */}
                        {selectedPost.comments.map((comment, index) => (
                            <Comment username={comment.username} comment={comment.content} spoiler={comment.isSpoiler}/>
                        ))}
                    </div>
                    <div className="message">
                        <input type="text" placeholder="Comment Here" value={comment} onChange={e => setComment(e.target.value)} onKeyDown={checkSubmission} />
                    </div>
                </div>}
                {!selectedPost && <div className="App-right"></div>}
            </div>
        </div>
    );
}

export default Home;