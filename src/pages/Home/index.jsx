import logo from '../../assets/logo.svg';
import search from '../../assets/search.svg';
import chat from '../../assets/chat-bubble.svg';
import './style.less';
import Comment from '../../components/Comment';
import LargePost from '../../components/LargePost';
import Post from '../../components/Post';
import {useState, useEffect} from 'react';

function Home() {
    return (
        <div className="App">
        <header className="App-header">
            <img src={logo} alt="logo"/>
            <p>Logout</p>
        </header>
            <div className="App-body">
                <div className="App-left">
                    <div className="search-create-div">
                        <div className="search">
                            <img src={search} alt="search"/>
                            <input type="text" placeholder="Search" />
                        </div>
                        <div className="create">
                            <img src={chat} alt="chat"/>
                            Create Post
                        </div>
                    </div>
                    <Post username="Rinsworth" title="Your Name"/>
            <Post username="sqirley" title="Haikyuu"/>
                </div>
                <div className="App-right">
                    <div className="post-comment-div">
                        <LargePost username="Rinsworth" title="Your Name" body="I liked Your Name" />
                        <div className="blank"></div>
                        <Comment username="sqirley" comment="that movie was cool"/>
                    </div>
                    <div className="message">
                        <input type="text" placeholder="Comment Here" />
                    </div>
                </div>
            </div>
        </div>
    );
}

export default Home;